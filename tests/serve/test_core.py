from parrot.serve.core import create_serve_core

from parrot.testing.get_configs import get_sample_core_config_path


def test_launch_core():
    config_path = get_sample_core_config_path("localhost_serve_core.json")
    core = create_serve_core(config_path)
    print(core)


def test_core_register_session():
    config_path = get_sample_core_config_path("localhost_serve_core.json")
    core = create_serve_core(config_path)
    core.register_session({})


if __name__ == "__main__":
    test_launch_core()
    test_core_register_session()
