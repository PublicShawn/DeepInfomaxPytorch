from config import ConfigParser
from pqcontroller import PQController

if __name__ == "__main__":
    config = ConfigParser()
    controller = PQController(config)
    controller.build_train()
    controller.train()