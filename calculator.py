from contextlib import contextmanager
import signal
import torch as th

# taken from
# https://stackoverflow.com/questions/492519/timeout-on-a-function-call
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            return eval(formula)
    except Exception as e:
        signal.alarm(0)
        print(f"Warning: Failed to eval {formula}, exception: {e}")
        return None


def use_calculator(sample):
    if "<<" not in sample:
        return None

    parts = sample.split("<<")
    remaining = parts[-1]
    if ">>" in remaining:
        return None
    if "=" not in remaining:
        return None
    lhs = remaining.split("=")[0]
    lhs = lhs.replace(",", "")
    if any([x not in "0123456789*+-/.()" for x in lhs]):
        return None
    return eval_with_timeout(lhs)


def batch_calculator_sample(model, qn, tokenizer, device, sample_len, **kwargs):
    EQUALS_TOKENS = set(tokenizer.convert_tokens_to_ids(["=", "Ġ=", ")="]))
    #  assert EQUALS_TOKENS == set([28, 796, 47505])
    #  EQUALS_TOKENS = set([28, 796, 47505])
    ANS_TOKEN = set([tokenizer.convert_tokens_to_ids("[ANS]")])

    model_kwargs = {}
    past_key_values = None
    generated_token_ids = [[] for _ in range(len(qn))]
    finished = [False] * len(qn) #初始化一个布尔列表，用于标记每个问题是否已完成
    current_patience = 0
    patience_after_all_finished = 11
    tokenizer.padding_side = "left"
    for _ in range(sample_len):   #在每次循环迭代中，模型使用上一个循环迭代生成的部分文本序列作为输入，并继续生成下一个token。这个过程一直持续到生成整个文本序列或达到指定的最大循环次数。循环的次数由 sample_len 控制。
        with th.no_grad():
            #使用分词器对问题进行编码，生成输入的编码信息。包括将问题转化为token ID并添加特殊标记
            inputs_encoding = tokenizer(      #{"input_ids":, "attention_mask":}
                qn,
                return_attention_mask=True,
                return_tensors="pt", 
                add_special_tokens=False,
                padding=True,
            ).to(device)
            #  attention_mask = th.where(inputs_encoding["input_ids"] == tokenizer.pad_token_id, 0, 1)
            #  inputs_encoding["attention_mask"] = attention_mask
            #  inputs_encoding = inputs_encoding.to(device)
            #  if _ == 0 or _ == sample_len - 1:
            #      print("inputs_encoding", inputs_encoding)

            #记录编码后的输入的原始长度
            orig_len = inputs_encoding["input_ids"].shape[1]

            if past_key_values and past_key_values[0][0].size(-2) != orig_len - 1:
                #  print("past key values size: ", past_key_values[0][0].size(-2))
                #  print("current input ids length: ", orig_len)
                past_key_values = None

            model_kwargs["past"] = past_key_values
            #使用模型生成解决方案的token ID\
            generated_tokens = model.generate(**inputs_encoding, max_length=orig_len + 1 , pad_token_id=tokenizer.pad_token_id, use_cache=True, **model_kwargs, **kwargs)  # generated_tokens length==4

            generated_tokens_length = len(generated_tokens)
            # print("Generated tokens length:", generated_tokens_length) 
            # out, model_outputs = generated_tokens

            # out, model_outputs = model.generate(
            out = model.generate(
                **inputs_encoding,
                max_length=orig_len + 1,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                **model_kwargs,
                **kwargs,
            )

            # past_key_values = model_outputs.past_key_values
            # past_key_values = model_kwargs.get("past", None)

            #将生成的token ID解码为文本解决方案
            text = tokenizer.batch_decode(out, skip_special_tokens=True)
            #  if _ == 0 or _ == sample_len - 1:
                #  print("out", out)
                #  print("text", text)
            for i in range(len(generated_token_ids)):
                generated_token_ids[i].append(out[i, -1].item()) #将在当前循环迭代中生成的最后一个 token 的 ID 添加到 generated_token_ids[i] 列表中
                #  if out[i, -1].item() in EQUALS_TOKENS:
                if generated_token_ids[i][-1] in EQUALS_TOKENS:
                    #检查是否生成了等号相关的token，如果生成了，会触发计算器函数 use_calculator 对结果进行计算，并将计算结果添加到文本中
                    answer = use_calculator(text[i])
                    #如果答案不为空（即成功计算出答案），则进入这个条件
                    if answer is not None: 
                        #  print("Triggered calculator, answer", answer)
                        #将计算的答案以字符串形式添加到 text[i] 中，后面跟着一个特殊标记 ">>
                        text[i] = text[i] + str(answer) + ">>"  
                        #  generated_token_ids[i].extend(tokenizer.convert_tokens_to_ids([str(answer), ">>"]))
                        generated_token_ids[i].extend(tokenizer(str(answer), ">>", add_special_tokens=False).input_ids)
                        past_key_values = None
                if generated_token_ids[i][-1] in ANS_TOKEN:
                    finished[i] = True
                    #  print(finished)

            qn = text

            if all(finished):
                current_patience += 1
                #  print("patience: ", current_patience)
                #  print(qn)
            if current_patience >= patience_after_all_finished:
                #  print("early stop because patience!")
                break

    tokenizer.padding_side = "right"

    return qn, generated_token_ids

