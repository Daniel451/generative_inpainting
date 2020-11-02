import asyncio

import cv2
import neuralgym as ng
import numpy as np
import tensorflow as tf
from grpclib.server import Server
from grpclib.utils import graceful_exit

from api.protodef import ganpaper_pb2_grpc
from inpaint_model import InpaintCAModel


class Servicer(ganpaper_pb2_grpc.AnonymizerServicer):

    def __init__(self):
        FLAGS = ng.Config('inpaint.yml')
        model = InpaintCAModel()
        # mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)


        h, w, _ = image.shape
        grid = 8
        image = image[:h // grid * grid, :w // grid * grid, :]
        mask = mask[:h // grid * grid, :w // grid * grid, :]
        print('Shape of image: {}'.format(image.shape))

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            input_image = tf.constant(input_image, dtype=tf.float32)
            output = model.build_server_graph(FLAGS, input_image)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)
            # load pretrained model
            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
                assign_ops.append(tf.assign(var, var_value))
            sess.run(assign_ops)
            print('Model loaded.')
            result = sess.run(output)
            cv2.imwrite(args.output, result[0][:, :, ::-1])


    async def GetAnonymizedImage(self, request, context):
        # original image
        image = np.frombuffer(request.original.data, np.uint8)
        image = image.reshape((request.original.height, request.original.width, 3))

        # mask
        mask = np.frombuffer(request.mask.data, np.uint8)
        mask = mask.reshape((request.mask.height, request.mask.width, 3))





async def serve(*, host="127.0.0.1", port=50051):
    server = Server([Servicer()])
    with graceful_exit([server]):
        await server.start(host, port)
        await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(serve())
