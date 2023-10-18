from Parrot.parrot.vm.controller import Controller
from Parrot.parrot.os.tokenizer import Tokenizer


def test_encode():
    ctrl = Controller()
    tokenized_storage = Tokenizer(ctrl)
    tokenizer = "hf-internal-testing/llama-tokenizer"
    ctrl.register_tokenizer(tokenizer)

    # prompt_text = "He is widely acknowledged as one of the top achievers in his class"
    prompt_text = "</s>"
    encoded = tokenized_storage.tokenize(prompt_text, tokenizer)

    print(encoded)


def test_decode():
    ctrl = Controller()
    tokenized_storage = Tokenizer(ctrl)
    tokenizer = "hf-internal-testing/llama-tokenizer"
    ctrl.register_tokenizer(tokenizer)

    # token_ids = [
    #     940,
    #     338,
    #     17644,
    #     24084,
    #     3192,
    #     408,
    #     697,
    #     310,
    #     278,
    #     2246,
    #     3657,
    #     347,
    #     874,
    #     297,
    #     670,
    #     770,
    # ]

    # token_ids = [29871]

    print(tokenized_storage.detokenize([310, 278], tokenizer))

    decoded = tokenized_storage.detokenize(token_ids[:9], tokenizer)
    decoded += tokenized_storage.detokenize(token_ids[8:], tokenizer)

    print(decoded)


if __name__ == "__main__":
    test_encode()
    # test_decode()
