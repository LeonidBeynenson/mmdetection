import pathlib
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy

import torch
from mmdet.utils import get_root_logger

try:
    import nncf

    _is_nncf_enabled = True
except:
    _is_nncf_enabled = False


def is_nncf_enabled():
    return _is_nncf_enabled


def check_nncf_is_enabled():
    if not is_nncf_enabled():
        raise RuntimeError("Tried to use NNCF, but NNCF is not installed")


if is_nncf_enabled():
    try:
        from nncf.initialization import InitializingDataLoader
        from nncf.structures import QuantizationRangeInitArgs
        from nncf.compression_method_api import CompressionAlgorithmController

        from nncf import NNCFConfig
        from nncf import load_state
        from nncf import create_compressed_model, register_default_init_args
        from nncf.utils import get_all_modules
        from nncf.dynamic_graph.context import no_nncf_trace as original_no_nncf_trace

        class_CompressionAlgorithmController = CompressionAlgorithmController
        class_InitializingDataLoader = InitializingDataLoader
    except:
        raise RuntimeError("Incompatible version of NNCF")
else:
    class DummyCompressionAlgorithmController:
        pass
    class DummyInitializingDataLoader:
        pass

    class_CompressionAlgorithmController = DummyCompressionAlgorithmController
    class_InitializingDataLoader = DummyInitializingDataLoader

def check_NNCFNetwork_is_compatible():
    check_nncf_is_enabled()
    from nncf.nncf_network import NNCFNetwork
    assert not hasattr(NNCFNetwork, "export"), "NNCFNetwork should not contain 'export', since it is defined in BaseDetector"

SHOULD_USE_DUMMY_FORWARD_WITH_EXPORT_PART = True
def wrap_nncf_model(model, cfg, data_loader_for_init=None, get_fake_input_func=None):
    """
    The function wraps mmdet model by NNCF
    Note that the parameter `get_fake_input_func` should be the function `get_fake_input`
    -- cannot import this function here explicitly
    """
    check_nncf_is_enabled()
    check_NNCFNetwork_is_compatible()
    pathlib.Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    nncf_config = NNCFConfig(cfg.nncf_config)
    logger = get_root_logger(cfg.log_level)

    if data_loader_for_init:
        wrapped_loader = MMInitializeDataLoader(data_loader_for_init)
        # TODO: [NNCF] need check the arguments in register_default_init_args()
        # TODO: add loss factory that reads config file, creates them and passes to register_default_init_args()
        nncf_config.register_extra_structs([QuantizationRangeInitArgs(wrapped_loader)])
    elif not cfg.nncf_load_from:
        raise RuntimeError("Tried to load NNCF checkpoint, but there is no path")

    if cfg.nncf_load_from:
        resuming_state_dict = load_checkpoint(model, cfg.nncf_load_from)
        logger.info(f"loaded nncf checkpoint from {cfg.nncf_load_from}")
    else:
        resuming_state_dict = None

    def __get_fake_data_for_forward_export(cfg, nncf_config, get_fake_input_func):
        # based on the method `export` of BaseDetector from mmdet/models/detectors/base.py
        # and on the script tools/export.py
        assert get_fake_input_func is not None

        input_size = nncf_config.get("input_info").get('sample_size')
        assert len(input_size) == 4 and input_size[0] == 1

        H, W = input_size[-2:]
        C = input_size[1]
        orig_img_shape = tuple([H, W, C]) #HWC order here for np.zeros to emulate cv2.imread

        device = next(model.parameters()).device

        # NB: the full cfg is required here!
        fake_data = get_fake_input_func(cfg, orig_img_shape=orig_img_shape, device=device)
        return fake_data
    def _get_fake_data_for_forward_export():
        # make a closure to use config and get_fake_input_func from the external scope
        cfg_copy = deepcopy(cfg)
        nncf_config = NNCFConfig(cfg_copy.nncf_config)
        return __get_fake_data_for_forward_export(cfg_copy, nncf_config, get_fake_input_func)

    def dummy_forward_without_export_part(model):
        input_size = nncf_config.get("input_info").get('sample_size')
        device = next(model.parameters()).device
        input_args = ([torch.randn(input_size).to(device), ],)
        input_kwargs = dict(return_loss=False, dummy_forward=True)
        model(*input_args, **input_kwargs)

    def dummy_forward_with_export_part(model):
        # based on the method `export` of BaseDetector from mmdet/models/detectors/base.py
        # and on the script tools/export.py
        fake_data = _get_fake_data_for_forward_export()
        img = fake_data["img"]
        img_metas = fake_data["img_metas"]
        with model.forward_export_context(img_metas):
            model(img)

    if SHOULD_USE_DUMMY_FORWARD_WITH_EXPORT_PART:
        dummy_forward = dummy_forward_with_export_part
    else:
        dummy_forward = dummy_forward_without_export_part

    model.dummy_forward_fn = dummy_forward

    compression_ctrl, model = create_compressed_model(model, nncf_config, dummy_forward_fn=dummy_forward,
                                                      resuming_state_dict=resuming_state_dict)
    print(*get_all_modules(model).keys(), sep="\n")
    return compression_ctrl, model


def load_checkpoint(model, filename, map_location=None, strict=False):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError('No state_dict found in checkpoint file {}'.format(filename))
    _ = load_state(model, state_dict, strict)
    return checkpoint


def export_model_to_onnx(compression_ctrl, f_name):
    logger = get_root_logger(cfg.log_level)
    logger.error("The function 'mmdet.core.nncf.export_model_to_onnx' is obsolete now "
                 "-- please, use the script tools/export.py with the same config file and the corresponding snapshot")
    logger.error("Now the function 'mmdet.core.nncf.export_model_to_onnx' does nothing and return")

class MMInitializeDataLoader(class_InitializingDataLoader):
    def get_inputs(self, dataloader_output):
        # redefined InitializingDataLoader because
        # of DataContainer format in mmdet
        kwargs = {k: v.data[0] for k, v in dataloader_output.items()}
        return (), kwargs

    # TODO: not tested; need to test
    def get_target(self, dataloader_output):
        return dataloader_output["gt_bboxes"], dataloader_output["gt_labels"]


@contextmanager
def nullcontext():
    """
    Context which does nothing; is needed to support python > python3.7
    """
    yield


def no_nncf_trace():
    """
    Wrapper for original NNCF no_nncf_trace() context
    """
    if is_nncf_enabled():
        return original_no_nncf_trace()
    return nullcontext()
