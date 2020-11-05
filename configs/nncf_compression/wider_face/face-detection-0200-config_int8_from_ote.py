_base_ = ['../../../../openvino_training_extensions/pytorch_toolkit/object_detection/model_templates/face-detection/face-detection-0200/model.py']
data = dict(
    samples_per_gpu=32,
)
# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0005)
# runtime settings
total_epochs = 3
log_level = 'INFO'
work_dir = './output'


load_from = 'https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/face-detection-0200.pth'

find_unused_parameters = True

nncf_config = dict(
    input_info=dict(
        sample_size=[1, 3, 256, 256]
    ),
    compression=[
        dict(
            algorithm='quantization',
            initializer=dict(
                range=dict(
                    num_init_steps=10
                ),
                batchnorm_adaptation=dict(
                    num_bn_adaptation_steps=30,
                )
            )
        )
    ],
    log_dir=work_dir
)
