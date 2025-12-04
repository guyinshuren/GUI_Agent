from hdc_utils import setup_device


def main(action):
    device = setup_device()
    device.step(action)


if __name__ == "__main__":
    Action = {"thought": "iiii", "TYPE": " "}
    main(Action)
