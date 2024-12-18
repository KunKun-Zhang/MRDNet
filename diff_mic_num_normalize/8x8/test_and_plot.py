from differentialNet import DifferentialNet1
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

if __name__ == '__main__':
    model = DifferentialNet1()
    model.load_state_dict(torch.load(
        'E:\\pyproject\\DOAEstimation\\DOA\\copy_model\\differential_net15.pth'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    time_series = np.load('E:\\pyproject\\DOAEstimation\\DOA\\copy_model\\signals.npy', allow_pickle=True)
    spectrum = fft(time_series)
    spectrum = spectrum[..., range(spectrum.shape[-1] // 2 + 1)]
    real = np.real(spectrum).astype(np.float32)
    imag = np.imag(spectrum).astype(np.float32)
    real_imag = np.concatenate((real, imag), axis=1)
    real_imag = np.transpose(real_imag, (1, 0))
    real_imag = np.reshape(real_imag, (real_imag.shape[0], 8, 8))
    real_imag = real_imag[None, ...]
    real_imag = torch.from_numpy(real_imag).to(device)
    out = model(real_imag)
    out = out.cpu().detach().numpy()[0]
    ttt = np.sum(out)
    plt.plot(out)
    plt.axvline(180, linestyle='--', color='r', label='180')
    plt.xlabel('angle(°)')
    plt.ylabel('probability')
    plt.legend()
    plt.show()

    with open('draw_data.txt', 'w') as f:
        for data in out:
            f.write(str(data))
            f.write('\n')
