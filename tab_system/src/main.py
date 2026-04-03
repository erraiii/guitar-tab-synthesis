from config import VIDEO_PATH
from fusion.tab_generator import TabGenerator


if __name__ == "__main__":

    tg = TabGenerator(VIDEO_PATH)
    tg.generate()
