from ImageToAscii import *
import cv2
from PIL import Image
import numpy as np
import logging
import time
import pyvips

# map vips formats to np dtypes
FORMAT_TO_DTYPE= {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

# map np dtypes to vips
DTYPE_TO_FORMAT = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}

class VideoToAscii:
    def __init__(self, filenameOut, video):
        logging.basicConfig(
            format='%(levelname)s\t\t%(asctime)s: %(name)s - %(message)s',
            level=logging.INFO
        )

        self.videoIn = video

        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        self.videoOut = cv2.VideoWriter(
            filenameOut, cv2.CAP_FFMPEG, fourcc,
            fps, size
        )

        self.imageToAscii = ImageToAscii(None)

    @staticmethod
    def cv2ToPillow(arrayImage, **kwargs):
        return Image.fromarray(arrayImage, **kwargs)

    def convert_image(self, image, v_width, v_height, textcolor=[0,0,0], backgroundcolor=[255,255,255], **kwargs):
        frame_shape = (v_width, v_height)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        self.imageToAscii.set_image(image)
        self.imageToAscii.resize_grayscaleArray_scale(kwargs['scale'])

        buff = time.time()

        # convert image to ascii
        self.imageToAscii.convert(kwargs["times_dublicate"],
                                  textcolor=textcolor,
                                  backgroundcolor=backgroundcolor,
                                  font=kwargs["font"],
                                  width=v_width, height=v_height, dpi=kwargs['dpi'])


        # make array from ascii image
        asciiImage = self.imageToAscii.get_asciiImage()
        shape = [asciiImage.height, asciiImage.width, asciiImage.bands]

        logger = logging.getLogger('convert frame')
        logger.info(f"shape ascii image: {shape}, shape origin frame: {frame_shape}")

        asciiImageToArray = asciiImage.write_to_memory()
        image = np.ndarray(buffer=asciiImageToArray,
                           shape=shape,
                           dtype=FORMAT_TO_DTYPE[asciiImage.format])
        image = cv2.resize(image, frame_shape)

        return image

    def checkPrefer(self, path, pos):
        frame_count = self.videoIn.get(cv2.CAP_PROP_FRAME_COUNT)

        width = int(self.videoIn.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.videoIn.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_shape = (width, height)
        print(frame_count)
        print(width)
        print(height)
        # check for valid frame number
        if pos >= 0 & pos <= frame_count:
            # set frame position
            print("lol")
            self.videoIn.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = self.videoIn.read()
        if ret:
            frame = self.convert_image(frame, width, height,
                times_dublicate=1, font='consolas',
                scale=0.25)
            cv2.imshow("image", frame)
        print(ret)
        video.release()

    def convert(self):
        origin_w = self.videoIn.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_count = int(self.videoIn.get(cv2.CAP_PROP_FRAME_COUNT))

        width = int(self.videoIn.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.videoIn.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_shape = (width, height)
        count = 0

        logger = logging.getLogger('convert video')
        try:
            logger.info(f"frame number: {frame_count}")
            for i in range(int(frame_count)):
                start_time = time.time()
                success, frame = self.videoIn.read()
                # frame = self.convert_image(frame, width, height,textcolor=[0,0,128],
                #                            backgroundcolor=[0,255,0],
                #                            times_dublicate=2, font='consolas',
                #                            scale=0.25, dpi=20)
                frame = self.convert_image(frame, width, height, textcolor=[0, 0, 0],
                                           backgroundcolor=[0, 128, 0],
                                           times_dublicate=2, font='consolas',
                                           scale=0.12, dpi=40)
                cv2.imwrite('text.png' ,frame)
                self.videoOut.write(frame)
                logger.info(f"{count}/{frame_count}")
                count += 1
        except Exception as exp:
            raise exp
        finally:
            self.videoIn.release()
            self.videoOut.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    start_time = time.time()
    #
    video = cv2.VideoCapture(r"C:\Users\Lenovo\Videos\Captures\neo_bullets.mp4")
    app = VideoToAscii("neo_bullets.mp4", video)
    # app.checkPrefer(100)
    app.convert()

    # app.convert()
    print(time.time() - start_time)

    # video = video
    # if video.isOpened():C:\Users\Lenovo\PycharmProjects\VideoToAscii\venv\VideoToAscii.py
    #     width = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    #     height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    #     # print(cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT) # 3, 4
    #
    #     # or
    #     width = video.get(3)  # float
    #     height = video.get(4)  # float
    #
    #     print('width, height:', width, height)
    #
    #     fps = video.get(cv2.CAP_PROP_FPS)
    #     print('fps:', fps)  # float
    #     # print(cv2.CAP_PROP_FPS) # 5
    #
    #     frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    #     print('frames count:', frame_count)  # float
    #     # print(cv2.CAP_PROP_FRAME_COUNT) # 7