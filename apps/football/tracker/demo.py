'''
@Time    : 2022/11/26 15:16
@Author  : leeguandon@gmail.com
'''
import ast
import paddle
import argparse
from apps.pipeline.utils import merge_cfg, print_arguments
from apps.pipeline.pipeline import Pipeline


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=r"E:\comprehensive_library\Xiaobao\apps\football\tracker\configs\football_mot.yml",
        help="Path of configure",
    )
    parser.add_argument(
        "--image_file", type=str, default=None, help="Path of image file.")
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Dir of image file, `image_file` has a higher priority.")
    parser.add_argument(
        "--video_file",
        type=str,
        default=r"E:\comprehensive_library\Xiaobao\data\test\football_clip_jinqiu.mp4",
        help="Path of video file, `video_file` or `camera_id` has a highest priority."
    )  # 可以使用rtsp推流地址
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Dir of video file, `video_file` has a higher priority.")
    parser.add_argument(
        "--model_dir", nargs='*', help="set model dir in pipeline")
    parser.add_argument(
        "--camera_id",
        type=int,
        default=-1,  # 用来预测的摄像头ID，默认为-1(表示不使用摄像头预测，可设置为：0 - (摄像头数目-1) )，预测过程中在可视化界面按q退出输出预测结果到：output/output.m
        help="device id of camera to predict.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory of output visualization files.")
    parser.add_argument(
        "--run_mode",
        type=str,
        default='paddle',  # 使用GPU时，默认为paddle, 可选（paddle/trt_fp32/trt_fp16/trt_int8）
        help="mode of running(paddle/trt_fp32/trt_fp16/trt_int8)")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU."
    )
    parser.add_argument(
        "--enable_mkldnn",
        type=ast.literal_eval,
        default=False,  # CPU预测中是否开启MKLDNN加速，默认为False
        help="Whether use mkldnn with CPU.")
    parser.add_argument(  # 设置cpu线程数，默认为1
        "--cpu_threads", type=int, default=1, help="Num of threads with CPU.")
    parser.add_argument(
        "--trt_min_shape", type=int, default=1, help="min_shape for TensorRT.")
    parser.add_argument(
        "--trt_max_shape",
        type=int,
        default=1280,
        help="max_shape for TensorRT.")
    parser.add_argument(
        "--trt_opt_shape",
        type=int,
        default=640,
        help="opt_shape for TensorRT.")
    parser.add_argument(
        "--trt_calib_mode",
        type=bool,
        default=False,  # tensorrt是否使用校准功能，默认为False，使用tensorrt的int8时，需设置为True，使用paddleslim量化后的模型时需要设置为False
        help="If the model is produced by TRT offline quantitative "
             "calibration, trt_calib_mode need to set True.")
    parser.add_argument(
        "--do_entrance_counting",
        action='store_true',  # 是否统计出入口流量
        help="Whether counting the numbers of identifiers entering "
             "or getting out from the entrance. Note that only support single-class MOT."
    )
    parser.add_argument(
        "--do_break_in_counting",
        action='store_true',  # 此项表示闯入区域检查
        help="Whether counting the numbers of identifiers break in "
             "the area. Note that only support single-class MOT and "
             "the video should be taken by a static camera.")
    parser.add_argument(
        "--region_type",
        type=str,
        default='horizontal',  # 'horizontal'（默认值）、'vertical'：表示流量统计方向选择；'custom'：表示设置闯入区域
        help="Area type for entrance counting or break in counting, 'horizontal' and "
             "'vertical' used when do entrance counting. 'custom' used when do break in counting. "
             "Note that only support single-class MOT, and the video should be taken by a static camera."
    )
    parser.add_argument(
        '--region_polygon',
        nargs='+',
        type=int,
        default=[],  # 设置闯入区域多边形多点的坐标，无默认值
        help="Clockwise point coords (x0,y0,x1,y1...) of polygon of area when "
             "do_break_in_counting. Note that only support single-class MOT and "
             "the video should be taken by a static camera.")
    parser.add_argument(
        "--secs_interval",
        type=int,
        default=2,
        help="The seconds interval to count after tracking")
    parser.add_argument(
        "--draw_center_traj",
        action='store_true',
        default=True,  # 是否绘制跟踪轨迹，默认为False
        help="Whether drawing the trajectory of center")
    parser.add_argument(
        "--speed_predict",
        action='store_true',
        help="Whether predicting the speed")
    parser.add_argument(
        "--mapping_ratio",
        nargs='+',
        type=float,
        default=[],
        help="The horizontal width of the camera pixel and the horizontal width of the field "
             "of view of the actual scene. (x,y) two values represent the actual "
             "transverse width respectively")
    parser.add_argument(
        "--x_ratio",
        nargs='+',
        type=float,
        default=[],
        help="X-axis segmented distance mapping, "
             "every group of three float (x1, x2, dis1) "
             "represents the actual distance mapped between x1 and x2")
    parser.add_argument(
        "--y_ratio",
        nargs='+',
        type=float,
        default=[],
        help="Y-axis segmented distance mapping, "
             "every group of three float (y1, y2, dis1) "
             "represents the actual distance mapped between y1 and y2")
    parser.add_argument(
        "--team_clas",
        nargs='+',
        type=str,
        # default=["blue", "basai", "white", "liufangzhe"],
        default=[],
        help="Color based team classification, "
             "receive four parameters(color1, name1, color2, name2),"
             "The optional color parameters are: [black, white, blue, red, yellow, green, purple, orange]")
    parser.add_argument(
        "--singleplayer",
        type=str,
        default="",
        help="using the single player mode, input a str as name of the player")
    parser.add_argument(
        "--boating",
        action='store_true',
        help="showing the angle of the paddle")
    parser.add_argument(
        "--ball_drawing",
        action='store_true',
        help="Draw the smooth curve of the ball")
    parser.add_argument(
        "--link_player",
        nargs='+',
        type=float,
        default=[],
        help="hightlight and link the player of given id")
    parser.add_argument(
        "--golf",
        type=bool,
        default=False,
        help="golf style analysis")
    parser.add_argument(
        "--player_recognize",
        type=bool,
        default=False,
        help="whether recognize number of player")
    parser.add_argument(
        "--ball_control",
        type=bool,
        default=False,
        help="whether display ball control team")
    parser.add_argument(
        "--show",
        type=bool,
        default=False,
        help="whether display real-time image")
    parser.add_argument(
        "--save_loc",
        type=bool,
        default=False,
        help="whether save ball location")
    parser.add_argument(
        "--loc_dir",
        type=str,
        default=None,
        help="Dir of location info file.")
    return parser


def main():
    cfg = merge_cfg(FLAGS)
    print_arguments(cfg)

    pipeline = Pipeline(FLAGS, cfg)
    pipeline.run()


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"

    main()
