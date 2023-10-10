import os
import jsonlines
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from gpt_modeling_base import GPT2BaseModel
from math_data_model import extract_answer, INVALID_ANS
from calculator import batch_calculator_sample as sample
from pysnooper import snoop
from torchsnooper import snoop as tsnoop

from pytorch_lightning.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from transformers.optimization import AdamW, get_scheduler
from torch.optim.lr_scheduler import LambdaLR
from typing import List
from pytorch_lightning.callbacks import BasePredictionWriter


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval="batch"):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices: List[int], 
        batch, batch_idx: int, dataloader_idx: int,
    ):
        with jsonlines.open(self.output_dir, 'a') as f:
            for p in prediction:
                f.write(p)


def get_custom_schedule(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return 1.
        return 0.1

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class GPT2ModelForGSM8K(GPT2BaseModel):
    """
    initiates a PyTorch Lightning GPT2 base model for training on GSM8K, defines training and evaluation steps
    """
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add GPT specific args
        Returns:
            parent_parser
        """
        parser = parent_parser.add_argument_group('GPT2ModelForGSM8K')
        parser.add_argument('--loss_on_prefix', action="store_true", default=False, help="Compute loss on question tokens")
        parser.add_argument('--prompt', action="store_true", default=False, help="Add chain of thought prompt before test question")
        parser.add_argument('--generator', action="store_true", default=True, help="Perform as a generator to generate solutions for training verifier")
        parser.add_argument('--temp', default=1.0, type=float, help="Temperature of generator when sampling")
        parser.add_argument('--num_sample', default=50, type=int, help="How many solutions to sample for each question")
        parser.add_argument('--sample_len', default=280, type=int, help="Maximum sample len")
        parser.add_argument('--comment', default=None, type=str, help="Comment for creating save dir")
        
        return parent_parser

    def __init__(self, args, model=None, tokenizer=None):
        super().__init__(args, model, tokenizer)

    def get_inputs(self, batch):  #按照batchsize读取数据
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'labels': batch['labels'],
        }

        return inputs   #inputs: {'input_ids': ,'attention_mask': ,'labels': }

    def training_step(self, batch, batch_idx):
        """ training step """
        inputs = self.get_inputs(batch)
        input_ids = inputs["input_ids"]  #torch.Size([4, 176])  seq_len : 176
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        #更新已处理的样本总数，这里考虑了数据并行时的情况
        self._consumed_samples += batch_size * max(self.trainer.gpus, 1)  # batch size * data parallel size
        labels = inputs["labels"]  #torch.Size([4, 176])
        self._consumed_tokens += len(labels.flatten()) * max(self.trainer.gpus, 1)

        #  loss, logits = self(**inputs)
        loss, logits = self(input_ids, inputs['attention_mask'], labels) #调用gpt_modeling_base.py的forward函数计算loss和预测结果logits
        ts_logger = self.logger.experiment

        if self.hparams.show_training_ex > -1 and batch_idx % self.hparams.show_training_ex == 0: #show_training_ex = -1 不打印训练例子
            #调用 show_training_example 方法展示训练示例的信息
            self.show_training_example(input_ids=input_ids[0], labels=labels[0], logits=logits[0])

        self.log("train_loss_step", loss, prog_bar=True, logger=True, on_step=True, batch_size=batch_size)
        #获取 TensorBoardLogger 的实例 ts_logger
        ts_logger = self.logger.experiment 
        #将训练损失与已处理的样本总数和标记总数关联起来，用于在 TensorBoard 中进行可视化
        ts_logger.add_scalar("train_loss_vs_samples", loss.item(), self._consumed_samples)
        ts_logger.add_scalar("train_loss_vs_tokens", loss.item(), self._consumed_tokens)

        # Do custom things for your task
        #执行自定义的训练步骤，并返回一个字典 custom_output_dict
        custom_output_dict = self.custom_training_step(batch, batch_idx, logits)

        output_dict = {"loss": loss}
        if custom_output_dict is not None:
            output_dict.update(custom_output_dict)
        #  current_step = self.trainer.lr_schedulers[0]['scheduler']._step_count

        return output_dict

    def custom_validation_step(self, batch, batch_idx, logits):
        batch_size = batch["input_ids"].size(0)
        return {'num_total': batch_size, 'question': batch['question'], 'answer': batch['answer']}

    def generate_step(self, batch, batch_idx):  ##batch:{"question": , "answer": , "question_id": }
        # 从批次中获取问题、答案和问题ID
        question = batch['question']
        answer = batch['answer']
        question_id = batch['question_id']
        solutions = []

        # 使用预定义的 sample 函数生成解决方案
        pred, generated_token_ids = sample(self.model, question, self.tokenizer, self.device, sample_len=self.hparams.sample_len, 
                                        do_sample=True, temperature=self.hparams.temp)
        generated_solutions = self.tokenizer.batch_decode(generated_token_ids)

        # 生成文件名并打开 JSONLines 文件以追加写入
        generator_file = "generator_solution.jsonl" + str(self.global_rank)
        with jsonlines.open(os.path.join(self.hparams.save_dir, self.hparams.timestamp + '-' + generator_file), 'a') as f:
            # 对生成的每个解决方案进行处理
            for idx, solution in enumerate(generated_solutions):
                # 提取预测答案和真实答案
                pred_answer = extract_answer(solution)
                gt_answer = extract_answer(answer[idx])
                
                # 确保真实答案不是无效答案
                assert gt_answer != INVALID_ANS
                
                
                # 添加解决方案到列表
                solutions.append(solution)
                
                # 将问题、真实答案、生成的解决方案等写入 JSONLines 文件
                f.write({"question": question[idx], "ground_truth": answer[idx],
                    "solution": solution, "is_correct": pred_answer == gt_answer, "question_id": question_id[idx]})
                
                # 如果预测答案和真实答案一致，则打印相关信息
                if pred_answer == gt_answer:
                    print('*' * 50)
                    print("question: ", question[idx])
                    print("predicted answer: ", solution)
                    print("gold answer: ", answer[idx])
                    print('*' * 50)



    def predict_step(self, batch, batch_idx):  #batch:{"question": , "answer": , "question_id": }
        """ batch calculator predict step """
        question = batch['question'] #'[QUES]....\n'
        for idx, q in enumerate(question): # 将问题中的每个文本添加一个特殊标记"[THOUGHT]"
            question[idx] += "[THOUGHT]"
        batch['question'] = question  #将添加了特殊标记的问题更新回批次数据中  "[QUES].......\n[THOUGHT]"
        answer = batch['answer']  #'[THOUGHT].....\n[ANS]...<|endoftext|>'
        question_id = batch['question_id']
        solutions = []
        is_correct = []

        if batch_idx % 100 == 0:
            print(f"Current predict example index: {batch_idx}.")
        if self.hparams.generator:
            #  solution_file = "generator_solution.jsonl" + str(self.global_rank)
            if self.hparams.data_name == "gsm8k":
                #调用calculator.py 生成solution    使用给定的模型、问题、分词器等参数进行采样，生成预测结果和生成的token ID  
                pred, generated_token_ids = sample(self.model, question, self.tokenizer, self.device, sample_len=self.hparams.sample_len, 
                                                do_sample=True, temperature=self.hparams.temp)
                #生成的token ID解码为文本解决方案
                generated_solutions = self.tokenizer.batch_decode(generated_token_ids)
            else:
                raise ValueError()
        else:
            #  solution_file = "model_solution.jsonl" + str(self.global_rank)
            pred, generated_token_ids = sample(self.model, question, self.tokenizer, self.device, sample_len=self.hparams.sample_len)
            generated_solutions = self.tokenizer.batch_decode(generated_token_ids)

        predictions = []
        for idx, solution in enumerate(generated_solutions):
            if self.hparams.data_name == "gsm8k":
                #提前答案
                pred_answer = extract_answer(solution)
                gt_answer = extract_answer(answer[idx])
            else:
                raise ValueError()
            # assert gt_answer != INVALID_ANS
            is_correct.append(int(pred_answer == gt_answer))
            solutions.append(solution)
            prediction = {"question": question[idx], "ground_truth": answer[idx], "solution": solution, 
                    "is_correct": pred_answer == gt_answer, "question_id": question_id[idx]}
            predictions.append(prediction)
            #  f.write({"question": question[idx], "ground_truth": answer[idx],
            #      "solution": solution, "is_correct": pred_answer == gt_answer, "question_id": question_id[idx]})
            if pred_answer == gt_answer:
                print('*' * 50)
                print("question: ", question[idx])
                print("predicted answer: ", solution)
                print("gold answer: ", answer[idx])
                print('*' * 50)

        return predictions

#配置一个回调函数，该回调函数将在训练过程中定期将训练结果写入到指定的输出目录中，以供后续分析和查看。
    def configure_callbacks(self):
        if self.hparams.generator:
            solution_file = "generator_solution.jsonl" + str(self.global_rank)
        else:
            solution_file = "model_solution.jsonl" + str(self.global_rank)
        #generator生成的solution路径
        output_dir = os.path.join(self.hparams.save_dir, self.hparams.timestamp + '-' + solution_file)  #'/raid/model/gpt2/829-generator_solution.jsonl0'

        return CustomWriter(output_dir=output_dir, write_interval="batch")

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_params = [
            {'params': [p for n, p in self.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.hparams.l2},
            {'params': [p for n, p in self.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # Configure optimizer.
        if isinstance(self.trainer.strategy, DeepSpeedStrategy):
            if 'offload_optimizer' in self.trainer.training_type_plugin.config['zero_optimization']:
                optimizer = DeepSpeedCPUAdam(
                    optimizer_grouped_params, adamw_mode=True,
                    lr=self.hparams.lr,
                    betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
                    #  eps=self.hparams.adam_epsilon,
                    )
            else:
                optimizer = FusedAdam(
                    optimizer_grouped_params, adam_w_mode=True,
                    lr=self.hparams.lr,
                    betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
                    #  eps=self.hparams.adam_epsilon,
                    )
        else:
            optimizer = AdamW(optimizer_grouped_params, lr=self.hparams.lr,
                              betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
                              #  eps=self.hparams.adam_epsilon,
                              )
        # Configure learning rate scheduler.
        warmup_steps = self.hparams.warmup * self.total_step
        # custom schedule
        #  scheduler = get_custom_schedule(optimizer=optimizer,
                #  num_warmup_steps=warmup_steps, num_training_steps=self.total_step)
        scheduler = get_scheduler(name=self.hparams.scheduler, optimizer=optimizer,
                                  num_warmup_steps=warmup_steps, num_training_steps=self.total_step)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [{
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }]

