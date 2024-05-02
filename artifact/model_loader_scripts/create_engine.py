from parrot.engine.engine_creator import create_engine
import sys

if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "7b":
        config_path = "engine_llama7b.json"
    elif arg == "13b":
        config_path = "engine_llama13b.json"
    else:
        raise ValueError("Invalid argument. Please use '7b' or '13b' as argument")

    engine = create_engine(
        engine_config_path=config_path,
        connect_to_os=False,
    )
