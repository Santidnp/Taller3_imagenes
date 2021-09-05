import cv2
import numpy as np

class Diezmado:
    
    def __init__(self,path,D):
        self.path = path
        self.D = D
        self.image = cv2.imread(path)

    def Filtrado(self):
        # high pass filter mask
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        image_gray_fft = np.fft.fft2(image_gray)
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

        num_rows, num_cols = (image_gray.shape[0], image_gray.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        half_size = num_rows / 2  # here we assume num_rows = num_columns

        high_pass_mask = np.zeros_like(image_gray)
        freq_cut_off = 1/int(self.D) # it should less than 1
        radius_cut_off = int(freq_cut_off * half_size)
        idx_hp = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) > radius_cut_off
        high_pass_mask[idx_hp] = 1
        mask = high_pass_mask  # can also use high or band pass mask
        fft_filtered = image_gray_fft_shift * mask
        self.image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        self.image_filtered = np.absolute(self.image_filtered)
        self.image_filtered /= np.max(self.image_filtered)

    def diezmado(self):
        image_decimated = self.image_filtered[::int(self.D), ::int(self.D)]
        cv2.imshow("Imagen Diezmada", image_decimated)
        cv2.waitKey(0)



imagen = Diezmado(r'C:\Users\sngh9\OneDrive\Escritorio\Maestria_Semestre_2\Procesamiento_de_imagenes\Taller_3\lena.png',3)
imagen.Filtrado()
imagen.diezmado()
