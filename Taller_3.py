#Elaborado por: Santiago Garcia, Laura Juliana Ramos
import cv2
import numpy as np


class filtrado:

    def __init__(self, path):
        self.path = path


    def descomposicion(self,N):
        image = cv2.imread(self.path)
        self.N=N
        image_gray1= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray3 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray4 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for i in range(N):

            kernelH= np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            kernelV = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            kernelD = np.array([[2, -1, -2], [-1, 4, -1], [-2, -1, 2]])
            kernelL = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])

            image_convolvedH = cv2.filter2D(image_gray1, -1, kernelH)
            image_convolvedV = cv2.filter2D(image_gray2, -1, kernelV)
            image_convolvedD = cv2.filter2D(image_gray3, -1, kernelD)
            image_convolvedL = cv2.filter2D(image_gray4, -1, kernelL)

            image_gray1=image_convolvedH
            image_gray2=image_convolvedV
            image_gray3=image_convolvedD
            image_gray4=image_convolvedL

            IH= self.diezmado(2,image_convolvedH)
            IV=self.diezmado(2,image_convolvedV)
            ID=self.diezmado(2,image_convolvedD)
            IL=self.diezmado(2,image_convolvedL)

    def diezmado(self,D,image):
        self.D=D
        # high pass filter mask
        image_gray_fft = np.fft.fft2(image)
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

        num_rows, num_cols = (image_gray.shape[0], image_gray.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        half_size = num_rows / 2  # here we assume num_rows = num_columns

        high_pass_mask = np.zeros_like(image_gray)
        freq_cut_off = 1 / int(self.D)  # it should less than 1
        radius_cut_off = int(freq_cut_off * half_size)
        idx_hp = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) < radius_cut_off
        high_pass_mask[idx_hp] = 1
        mask = high_pass_mask  # can also use high or band pass mask
        fft_filtered = image_gray_fft_shift * mask
        self.image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        self.image_filtered = np.absolute(self.image_filtered)
        self.image_filtered /= np.max(self.image_filtered)

        image_decimated = self.image_filtered[::int(self.D), ::int(self.D)]
        cv2.imshow("Imagen Diezmada", image_decimated)
        cv2.waitKey(0)

    def interpolado(self,I):
        image = cv2.imread(self.path)
        self.I=I
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray_fft = np.fft.fft2(image_gray)
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

        num_rows, num_cols = (image_gray.shape[0], image_gray.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        half_size = num_rows / 2  # here we assume num_rows = num_columns

        high_pass_mask = np.zeros_like(image_gray)
        freq_cut_off = 1 / int(self.I)  # it should less than 1
        radius_cut_off = int(freq_cut_off * half_size)
        idx_hp = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) < radius_cut_off
        high_pass_mask[idx_hp] = 1
        mask = high_pass_mask  # can also use high or band pass mask
        fft_filtered = image_gray_fft_shift * mask
        self.image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        self.image_filtered = np.absolute(self.image_filtered)
        self.image_filtered /= np.max(self.image_filtered)

        rows, cols = self.image_filtered.shape
        num_of_zeros = self.I-1
        image_zeros = np.zeros((num_of_zeros * rows, num_of_zeros * cols), dtype=self.image_filtered.dtype)
        image_zeros[::num_of_zeros, ::num_of_zeros] = self.image_filtered
        W = 2 * num_of_zeros + 1
        # filtering
        image_interpolated = cv2.GaussianBlur(image_zeros, (W, W), 0)
        image_interpolated *= num_of_zeros ** 2
        cv2.imshow("Imagen interpolada", image_interpolated)
        cv2.waitKey(0)


imagen = filtrado(r'C:\Users\Laura\Desktop\IMAGENES\lena.png')
imagen2=cv2.imread(r'C:\Users\Laura\Desktop\IMAGENES\lena.png')
image_gray = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)
imagen.diezmado(3,image_gray)
imagen.interpolado(2)
imagen.descomposicion(4)
