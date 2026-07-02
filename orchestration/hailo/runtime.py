import numpy as np


class HailoModel:
    def __init__(self, hef_path):
        self.hef_path = str(hef_path)

    def __enter__(self):
        from hailo_platform import (  # type: ignore
            ConfigureParams,
            FormatType,
            HailoStreamInterface,
            HEF,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
            VDevice,
        )

        self.hef = HEF(self.hef_path)
        self.device = VDevice()
        self.target = self.device.__enter__()
        params = ConfigureParams.create_from_hef(
            self.hef,
            interface=HailoStreamInterface.PCIe,
        )
        self.network_group = self.target.configure(self.hef, params)[0]
        self.network_params = self.network_group.create_params()

        input_info = self.hef.get_input_vstream_infos()[0]
        output_info = self.hef.get_output_vstream_infos()[0]
        self.input_name = input_info.name
        self.output_name = output_info.name

        input_params = InputVStreamParams.make(
            self.network_group,
            format_type=FormatType.FLOAT32,
        )
        output_params = OutputVStreamParams.make(
            self.network_group,
            format_type=FormatType.FLOAT32,
        )

        self.activation = self.network_group.activate(self.network_params)
        self.activation.__enter__()
        self.streams = InferVStreams(
            self.network_group,
            input_params,
            output_params,
        )
        self.pipeline = self.streams.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.streams.__exit__(exc_type, exc, traceback)
        self.activation.__exit__(exc_type, exc, traceback)
        self.device.__exit__(exc_type, exc, traceback)

    def infer(self, batch):
        output = self.pipeline.infer({self.input_name: batch.astype(np.float32)})
        return np.asarray(output[self.output_name])
