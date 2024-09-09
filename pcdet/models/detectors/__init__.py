from .detector3d_template import Detector3DTemplate
from .pointpillar import PointPillar
from .second_net import SECONDNet
from .centerpoint import CenterPoint
from .ssd3d import SSD3D
from .oprnet import oprnet
from .axuiliary_2d import Aux_2d

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PointPillar': PointPillar,
    'CenterPoint': CenterPoint,
    'oprnet': oprnet,
    'SSD3D': SSD3D,
    'oprnet': oprnet,
    'axuiliary_2d': Aux_2d
}


def build_detector(model_cfg, num_class, dataset, logger):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset, logger=logger
    )

    return model
