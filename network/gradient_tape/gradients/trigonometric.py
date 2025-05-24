import numpy as np

class TrigonometricGradients:
    @staticmethod
    def sin(grad_output, inputs):
        z = inputs[0]
        grad_output_h, *_ = grad_output if isinstance(grad_output, tuple) else (grad_output,)
        grad_h = grad_output_h * np.cos(z)
        return [grad_h]

    @staticmethod
    def cos(grad_output, inputs):
        z = inputs[0]
        grad_output_h, *_ = grad_output if isinstance(grad_output, tuple) else (grad_output,)
        dz = grad_output_h * -np.sin(z)
        return [dz]

    @staticmethod
    def tan(grad_output, inputs):
        z = inputs[0]
        grad_output_h, *_ = grad_output if isinstance(grad_output, tuple) else (grad_output,)
        dz = grad_output_h * (1 / np.cos(z) ** 2)
        return [dz]

    @staticmethod
    def arcsin(grad_output, inputs):
        z = inputs[0]
        grad_output_h, *_ = grad_output if isinstance(grad_output, tuple) else (grad_output,)
        dz = grad_output_h / np.sqrt(1 - z**2)
        return [dz]

    @staticmethod
    def arccos(grad_output, inputs):
        z = inputs[0]
        grad_output_h, *_ = grad_output if isinstance(grad_output, tuple) else (grad_output,)
        dz = -grad_output_h / np.sqrt(1 - z**2)
        return [dz]

    @staticmethod
    def arctan(grad_output, inputs):
        z = inputs[0]
        grad_output_h, *_ = grad_output if isinstance(grad_output, tuple) else (grad_output,)
        dz = grad_output_h / (1 + z**2)
        return [dz]

    @staticmethod
    def arctan2(grad_output, inputs):
        y, x = inputs
        grad_out_h, *_ = grad_output if isinstance(grad_output, tuple) else (grad_output,)
        w = x + 1j * y
        inv_w = 1.0 / w
        grad_y_h = grad_out_h * np.real(inv_w)
        grad_x_h = grad_out_h * np.imag(inv_w)
        return [grad_y_h, grad_x_h]

    @staticmethod
    def sinc(grad_output, inputs):
        x = inputs[0]
        grad_output_h, *_ = grad_output if isinstance(grad_output, tuple) else (grad_output,)
        pi_x = np.pi * x
        grad_val = np.where(x != 0,
                            (np.cos(pi_x) * np.pi * x - np.sin(pi_x)) / (np.pi * x**2),
                            0)
        grad_x_h = grad_output_h * grad_val
        return [grad_x_h]

    @staticmethod
    def hypot(grad_output, inputs):
        z1, z2 = inputs
        grad_output_h, grad_output_ah = grad_output if isinstance(grad_output, tuple) else (grad_output, np.zeros_like(z1))
        norm_sq = np.abs(z1)**2 + np.abs(z2)**2 + np.finfo(z1.dtype).eps
        r = np.sqrt(norm_sq)
        grad_z1_h  = grad_output_h  * (np.conj(z1) / (2 * r))
        grad_z1_ah = grad_output_ah * (z1 / (2 * r))
        grad_z2_h  = grad_output_h  * (np.conj(z2) / (2 * r))
        grad_z2_ah = grad_output_ah * (z2 / (2 * r))
        return [
            (grad_z1_h, grad_z1_ah),
            (grad_z2_h, grad_z2_ah)
        ]