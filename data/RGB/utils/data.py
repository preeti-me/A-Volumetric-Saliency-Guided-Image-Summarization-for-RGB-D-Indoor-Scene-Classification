import numpy as np
from scipy.linalg import toeplitz, cholesky, solve, solve_triangular, svd
from scipy.stats import gamma
from scipy.fft import fft, fftshift

def generate_signals(
    n_samples,
    Np=16,
    n_range_cells=1,
    SNR=15,
    texture_type="gaussian",
    texture_params=(1, 1),
    Tr=0.5e-1,
    include_target=True,
    rho=0.5,
    target_doppler=3,
    add_white_noise=False,
):

    signal = np.zeros((n_samples, Np, n_range_cells), dtype=np.complex64)
    signals_fft = np.zeros((n_samples, Np, n_range_cells), dtype=np.complex64)
    # Calculate the normalized Doppler frequency for the target
    # Construct the identity matrix
    identity_matrix = np.eye(Np)
    # Build the covariance matrix 'T' for the clutter, Create the Toeplitz covariance matrix
    T = toeplitz(rho ** np.arange(Np))
    if add_white_noise:
        # Total covariance matrix includes both clutter and white noise
        T_total = T + identity_matrix
    else:
        T_total = T
    # Cholesky decomposition of the total covariance matrix
    chol_T = cholesky(T)
    

    for n in range(n_samples):
        gaussian_noise = (np.random.randn(Np, n_range_cells) + 1j*np.random.randn(Np, n_range_cells)) / np.sqrt(2)
        if texture_type == "gamma":
            texture = gamma(texture_params[0], 0, texture_params[1]).rvs(size=n_range_cells)
            signal[n,:,:] = np.sqrt(texture) * (chol_T @ gaussian_noise)
        else:
            texture = np.ones(n_range_cells)
            signal[n,:,:] = (chol_T @ gaussian_noise)

        if add_white_noise:
            white_noise = (np.random.randn(Np, n_range_cells) + 1j * np.random.randn(Np, n_range_cells)) / np.sqrt(2)
            signal[n,:,:] += white_noise

    # Calculate the amplitude of the target signal based on the desired SNR
    if include_target:
        k = np.arange(Np)
        f_doppler = (target_doppler - Np // 2) / Np / Tr  # Normalized Doppler frequency
        # Generate the target signal vector 'p'
        p = np.exp(1j * 2 * np.pi * k * f_doppler * Tr) / np.sqrt(Np)
        target_amplitude = np.sqrt(10 ** (SNR / 10)) / np.sqrt(np.abs(p.conj().T @ np.linalg.solve(T_total, p)))
        target_signal = target_amplitude * p
        for n in range(n_samples):
            signal[n,:, 0] += target_signal
    # Perform Doppler processing (FFT over pulses)
    signal_fft = (1 / np.sqrt(Np)) * fft(signal, axis=1)
    # Shift zero-frequency component to the center of the spectrum
    signal_fft = fftshift(signal_fft, axes=1)

    return signal, signal_fft, T_total
   

def whiten_data(data, cov_matrix, reg_factor=1e-9, method='svd'):
    """
    Parameters:
    -----------
    data : ndarray (n_features, n_samples)
    cov_matrix : ndarray (n_features, n_features)
    reg_factor : float (défaut: 1e-6)
        Régularisation basée sur la trace: reg_factor * np.trace(cov_matrix)/n_features
    method : {'auto', 'cholesky', 'svd'}
    
    Returns:
    --------
    whitened_data : ndarray (n_features, n_samples)
    """
    # Régularisation adaptative (plus robuste)
    n_features = cov_matrix.shape[0]
    reg = reg_factor * np.trace(cov_matrix) / n_features
    
    # Matrice régularisée
    reg_cov = cov_matrix + reg * np.eye(n_features)
    
    if method == 'auto':
        try:
            # Cholesky (plus rapide si possible)
            L = cholesky(reg_cov, lower=True)
            L_inv = solve_triangular(L, np.eye(n_features), lower=True)
            return L_inv @ data
        except np.linalg.LinAlgError:
            method = 'svd'  # Fallback
    
    if method == 'svd':
        # SVD (toujours stable mais plus lent)
        U, s, _ = svd(reg_cov)
        s_inv_sqrt = 1.0 / np.sqrt(s + reg)
        whitening_matrix = U @ np.diag(s_inv_sqrt) @ U.T.conj()
        return whitening_matrix @ data
    
    raise ValueError("Méthode invalide. Choisir 'auto', 'cholesky' ou 'svd'")

def add_target(data, target_doppler, SNR, T_total, Tr=0.5e-1):
    Np = len(data)
    k = np.arange(Np)
    f_doppler = (target_doppler - Np // 2) / Np / Tr # Normalized Doppler frequency
    # Generate the target signal vector 'p'
    p = np.exp(1j * 2 * np.pi * k * f_doppler * Tr) / np.sqrt(Np)
    target_amplitude = np.sqrt(10 ** (SNR / 10)) / np.sqrt(np.abs(p.conj().T @ solve(T_total, p)))
    target_signal = target_amplitude * p
    data += target_signal
    return data

def blanchiment(X, texture, wn):
    bruit_ref = generate_signals(len(X), Np=16, n_range_cells=16, SNR=15, texture_type=texture,\
                        texture_params=(1, 1), Tr=0.5e-1, include_target=False, rho=0.5,\
                            target_doppler=0, add_white_noise=wn)[0]
    for i in range(len(X)):
        X[i] = whiten_data(X[i], bruit_ref[i])
    X = (1 / np.sqrt(len(X[0]))) * fft(X, axis=1)
    X = fftshift(X, axes=1)
    return X

def blanchiment_cible(X, texture, wn, doppler, snr, T):
    bruit_ref = generate_signals(len(X), Np=16, n_range_cells=16, SNR=15, texture_type=texture,\
                        texture_params=(1, 1), Tr=0.5e-1, include_target=False, rho=0.5,\
                            target_doppler=0, add_white_noise=wn)[0]
    for i in range(len(X)):
        X[i] = add_target(X[i], doppler, snr, T)
        X[i] = whiten_data(X[i], bruit_ref[i])
    X = (1 / np.sqrt(len(X[0]))) * fft(X, axis=1)
    X = fftshift(X, axes=1)
    return X

def blanchiment_doppler(X, texture, wn):
    bruit_ref_fft = generate_signals(len(X), Np=16, n_range_cells=16, SNR=15, texture_type=texture,\
                        texture_params=(1, 1), Tr=0.5e-1, include_target=False, rho=0.5,\
                            target_doppler=0, add_white_noise=wn)[1]
    for i in range(len(X)):
        X[i] = whiten_data(X[i], bruit_ref_fft[i])
    return X


def blanchiment_doppler_cible(X, texture, wn, doppler, snr, T):
    bruit_ref_fft = generate_signals(len(X), Np=16, n_range_cells=16, SNR=snr, texture_type=texture,\
                        texture_params=(1, 1), Tr=0.5e-1, include_target=False, rho=0.5,\
                            target_doppler=0, add_white_noise=wn)[1]
    for i in range(len(X)):
         X[i] = add_target(X[i], doppler, snr, T)
    X = (1 / np.sqrt(len(X[0]))) * fft(X, axis=1)
    X = fftshift(X, axes=1)
    for i in range(len(X)):
        X[i] = whiten_data(X[i], bruit_ref_fft[i])
    return X

def input_dsvdd(X, space):
    if (space == 'complex') or (space == 'vae'):
        return np.stack((X.real, X.imag), axis=1)
    if space == 'abs_comp':
        return np.abs(np.stack((X.real, X.imag), axis=1)) 
    if space == 'abs_comp2':
        return np.abs(np.stack((X.real, X.imag), axis=1))**2
    if space == 'exp':
        return np.exp(np.abs(np.stack((X.real, X.imag), axis=1)))
    if space == 'mod':
        return np.abs(X)
    if space == 'mod2':
        return np.abs(X)**2            
    else:
        print('space not mentionned')
        return np.stack((X.real, X.imag), axis=1)

 
def generate_signals_old(
    n_samples,
    Np=16,
    n_range_cells=1,
    SNR=15,
    texture_type="gaussian",
    texture_params=(1, 1),
    Tr=0.5e-1,
    include_target=True,
    output_type='complex',
    doppler_process=True,
    rho=0.5,
    target_doppler=3,
    target_range_cell=0,
    add_white_noise=False
):
    """
    Generate Range-Doppler maps with or without a target, supporting various clutter models.

    This function can generate:
    - Correlated Gaussian clutter
    - Correlated Gaussian clutter + white Gaussian noise
    - Correlated Compound Gaussian clutter
    - Correlated Compound Gaussian clutter + white Gaussian noise

    By setting 'rho' to 0, you can obtain the uncorrelated (white) cases.

    Parameters:
    ----------
    n_samples : int
        Number of Range-Doppler maps to generate.
    Np : int, optional
        Number of pulses (slow time samples). Default is 32.
    n_range_cells : int, optional
        Number of range cells. Default is 1.
    SNR : float, optional
        Signal-to-noise ratio in dB. Default is 10.
    texture_type : str or None, optional
        Type of texture distribution ('gamma') for compound Gaussian clutter.
        Set to None for standard Gaussian clutter. Default is None.
    texture_params : tuple, optional
        Parameters for the texture distribution. Default is (1, 1).
    Tr : float, optional
        Pulse repetition interval. Default is 0.5e-1.
    include_target : bool, optional
        Whether to include a target in the signal. Default is True.
    output_type : str, optional
        'complex' or 'amplitude' for the output signal type. Default is 'complex'.
    doppler_process : bool, optional
        Whether to perform Doppler processing (FFT over pulses). Default is False.
    correlated_noise : bool, optional
        Whether to generate correlated clutter. Default is False.
    rho : float, optional
        Correlation coefficient for the clutter (0 ≤ rho ≤ 1). Default is 0.5.
    target_doppler : int, optional
        Doppler bin index for the target (0 ≤ target_doppler < Np). Default is 3.
    target_range_cell : int, optional
        Range cell index for the target (0 ≤ target_range_cell < n_range_cells). Default is 0.
    add_white_noise : bool, optional
        Whether to add white Gaussian noise in addition to the clutter. Default is False.

    Returns:
    -------
    signals : ndarray
        Generated signals of shape (n_samples, Np, n_range_cells).
    """
    # Time indices for the pulses
    k = np.arange(Np)

    # Initialize the output array
    signals = np.zeros((n_samples, Np, n_range_cells), dtype=np.complex64 if output_type == 'complex' else np.float64)

    # Calculate the normalized Doppler frequency for the target
    f_doppler = (target_doppler - Np // 2) / Np / Tr  # Normalized Doppler frequency

    # Generate the target signal vector 'p'
    p = np.exp(1j * 2 * np.pi * k * f_doppler * Tr) / np.sqrt(Np)

    # Construct the identity matrix
    identity_matrix = np.eye(Np)

    # Build the covariance matrix 'T' for the clutter
    # Create the Toeplitz covariance matrix
    T = toeplitz(rho ** np.arange(Np))

    if add_white_noise:
        # Total covariance matrix includes both clutter and white noise
        T_total = T + identity_matrix
    else:
        T_total = T

    # Cholesky decomposition of the total covariance matrix
    chol_T = cholesky(T)

    # Calculate the amplitude of the target signal based on the desired SNR
    target_amplitude = np.sqrt(10 ** (SNR / 10)) / np.sqrt(np.abs(p.conj().T @ np.linalg.solve(T_total, p)))
    target_signal = target_amplitude * p
    # print(target_signal)
    # quit()

    # Initialize the texture distribution for compound Gaussian clutter
    if texture_type == "gamma":
        texture_distribution = gamma(*texture_params)
    elif texture_type == "gaussian":
        texture_distribution = "gaussian"
    else:
        raise ValueError("Unsupported texture type.")

    for i in range(n_samples):
        # Generate complex Gaussian noise (clutter)
        gaussian_noise = (np.random.randn(Np, n_range_cells) + 1j * np.random.randn(Np, n_range_cells)) / np.sqrt(2)
        # Apply the Cholesky decomposition to introduce correlation
        clutter = chol_T @ gaussian_noise

        if texture_distribution != "gaussian":
            # Generate texture samples for compound Gaussian clutter
            texture_samples = texture_distribution.rvs((Np, n_range_cells))
            texture_sqrt = np.sqrt(texture_samples)
            # Apply texture to the Gaussian noise
            clutter *= texture_sqrt
        else:
            # Standard Gaussian clutter
            clutter = clutter

        # Apply the Cholesky decomposition to introduce correlation
        # clutter = chol_T.T @ clutter

        if add_white_noise:
            # Add white Gaussian noise
            white_noise = (np.random.randn(Np, n_range_cells) + 1j * np.random.randn(Np, n_range_cells)) / np.sqrt(2)
            clutter += white_noise

        signal = clutter

        if include_target:
            # Add the target signal at the specified range cell
            signal[:, target_range_cell] += target_signal

        if doppler_process:
            # Perform Doppler processing (FFT over pulses)
            signal = (1 / np.sqrt(Np)) * fft(signal, axis=0)

        # Shift zero-frequency component to the center of the spectrum
        signal = fftshift(signal, axes=0)

        if output_type == 'amplitude':
            # Convert to amplitude squared
            signal = np.abs(signal) ** 2

        signals[i] = signal

    return signals

