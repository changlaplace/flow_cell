"""Differentiable propagation and phase-only optical components."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


def angular_spectrum_propagation(
    field: torch.Tensor,
    wavelength: float,
    dx: float,
    distance: float,
) -> torch.Tensor:
    """Propagate a complex field with the scalar Angular Spectrum Method."""
    height, width = field.shape[-2:]
    device = field.device
    dtype = field.real.dtype

    fy = torch.fft.fftfreq(height, d=dx, device=device, dtype=dtype)
    fx = torch.fft.fftfreq(width, d=dx, device=device, dtype=dtype)
    freq_y, freq_x = torch.meshgrid(fy, fx, indexing="ij")

    wave_number = 2 * math.pi / wavelength
    kx = 2 * math.pi * freq_x
    ky = 2 * math.pi * freq_y
    kz_squared = wave_number**2 - kx**2 - ky**2
    propagating = kz_squared >= 0
    kz = torch.sqrt(torch.clamp(kz_squared, min=0))

    transfer_function = torch.exp(1j * kz * distance) * propagating
    return torch.fft.ifft2(torch.fft.fft2(field) * transfer_function)


def pool_sensor(intensity: torch.Tensor, size: int) -> torch.Tensor:
    """Average-pool high-resolution intensity to a square detector."""
    return F.adaptive_avg_pool2d(
        intensity.unsqueeze(1), (size, size)
    ).squeeze(1)


class PhaseScreen(nn.Module):
    """Continuous phase-only screen with a low-resolution control grid."""

    def __init__(self, control_size: int):
        super().__init__()
        self.raw = nn.Parameter(
            0.05 * torch.randn(1, 1, control_size, control_size)
        )

    def phase(self, size: int) -> torch.Tensor:
        phase = F.interpolate(
            self.raw,
            size=(size, size),
            mode="bicubic",
            align_corners=False,
        )[0, 0]
        return math.pi * torch.tanh(phase)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return field * torch.exp(1j * self.phase(field.shape[-1]))


class OpticalStation(nn.Module):
    """One paired M1-cell-M2 transform with an intensity detector."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.m1 = PhaseScreen(config.phase_ctrl_N)
        self.m2 = PhaseScreen(config.phase_ctrl_N)

    def forward(
        self, cell_phase: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        field = torch.ones_like(cell_phase, dtype=torch.complex64)

        field = self.m1(field)
        field = angular_spectrum_propagation(
            field, config.wavelength_m, config.dx_m, config.z_m1_cell_m
        )
        field = field * torch.exp(1j * cell_phase)
        field = angular_spectrum_propagation(
            field, config.wavelength_m, config.dx_m, config.z_cell_m2_m
        )
        field = self.m2(field)
        field = angular_spectrum_propagation(
            field, config.wavelength_m, config.dx_m, config.z_m2_sensor_m
        )

        intensity = field.real.square() + field.imag.square()
        detector = pool_sensor(intensity, config.sensor_N)
        detector = detector / (
            detector.mean(dim=(-2, -1), keepdim=True) + 1e-8
        )
        return detector, intensity


asm = angular_spectrum_propagation
