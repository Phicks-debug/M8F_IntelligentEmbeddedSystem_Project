import numpy as np


class HailoDevice:
    def __enter__(self):
        from hailo_platform import VDevice  # type: ignore

        self.device = VDevice()
        self.device.__enter__()
        return self.device

    def __exit__(self, exc_type, exc, traceback):
        self.device.__exit__(exc_type, exc, traceback)


class HailoModel:
    def __init__(self, hef_path, device=None):
        self.hef_path = str(hef_path)
        self.device = device
        self.owns_device = device is None

    def __enter__(self):
        from hailo_platform import FormatType, HEF, VDevice  # type: ignore

        hef = HEF(self.hef_path)
        if self.owns_device:
            self.device = VDevice()
            self.device.__enter__()

        input_info = hef.get_input_vstream_infos()[0]
        output_infos = hef.get_output_vstream_infos()
        self.input_name = input_info.name
        self.output_names = [info.name for info in output_infos]
        self.output_shapes = {
            info.name: tuple(info.shape)
            for info in output_infos
        }

        self.model = self.device.create_infer_model(self.hef_path)
        self.model.input().set_format_type(FormatType.FLOAT32)
        for output_name in self.output_names:
            self.model.output(output_name).set_format_type(FormatType.FLOAT32)

        self.configured_model = self.model.configure()
        self.configured_model.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.configured_model.__exit__(exc_type, exc, traceback)
        if self.owns_device:
            self.device.__exit__(exc_type, exc, traceback)

    def infer(self, batch):
        bindings = self.configured_model.create_bindings()
        bindings.input().set_buffer(batch.astype(np.float32))

        outputs = {}
        for output_name in self.output_names:
            output = np.empty(self.output_shapes[output_name], dtype=np.float32)
            bindings.output(output_name).set_buffer(output)
            outputs[output_name] = output

        self.configured_model.run([bindings], timeout_ms=10000)
        if len(outputs) == 1:
            return next(iter(outputs.values()))
        return outputs
