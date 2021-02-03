from config import ConfigParser
from controller import Controller

if __name__ == "__main__":
    config = ConfigParser()
    controller = Controller(config)
    controller.build_train()
    controller.train()
    controller.build_test()
    controller.test()