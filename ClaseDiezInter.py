import cv2
import numpy as np


class Cambiotamano:

    def __init__(self,path):

        self.path = path
        self.imagen = cv2.imread(path)





    def Diezmado(self,D,img=None):

        self.D = D
        if img is None:

            imgris,half_size,high_pass_mask,col_iter, row_iter,num_rows, num_cols  = self.infoGris(self.imagen)
            print(imgris)

        else:
            imgris,half_size,high_pass_mask,col_iter, row_iter,num_rows, num_cols  = self.infoGris(img)

        image_gray_fft = np.fft.fft2(imgris)
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)
        freq_cut_off = 1 / int(self.D)
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



    def interpolacion(self,I,img=None):

        self.I = I

        if img is None:

            imgris, half_size, high_pass_mask, col_iter, row_iter,num_rows, num_cols  = self.infoGris(self.imagen)
            print(imgris)

        else:
            imgris, half_size, high_pass_mask, col_iter, row_iter,num_rows, num_cols  = self.infoGris(img)

        image_gray_fft = np.fft.fft2(imgris)
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)
        freq_cut_off = 1 / int(self.I)  # it should less than 1
        radius_cut_off = int(freq_cut_off * half_size)
        idx_hp = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) < radius_cut_off
        high_pass_mask[idx_hp] = 1
        mask = high_pass_mask  # can also use high or band pass mask
        fft_filtered = image_gray_fft_shift * mask
        self.image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        self.image_filtered = np.absolute(self.image_filtered)
        self.image_filtered /= np.max(self.image_filtered)


        num_of_zeros = self.I - 1
        image_zeros = np.zeros((num_of_zeros * num_rows, num_of_zeros * num_cols), dtype=self.image_filtered.dtype)
        image_zeros[::num_of_zeros, ::num_of_zeros] = self.image_filtered
        W = 2 * num_of_zeros + 1
        # filtering
        image_interpolated = cv2.GaussianBlur(image_zeros, (W, W), 0)
        image_interpolated *= num_of_zeros ** 2
        cv2.imshow("Imagen interpolada", image_interpolated)
        cv2.waitKey(0)

    def descomposicion(self, N,image=None):

        self.N = N
        if image is None:
            image = self.imagen

        image_gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray3 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray4 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for i in range(N):
            kernelH = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            kernelV = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            kernelD = np.array([[2, -1, -2], [-1, 4, -1], [-2, -1, 2]])
            kernelL = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])

            image_convolvedH = cv2.filter2D(image_gray1, -1, kernelH)
            image_convolvedV = cv2.filter2D(image_gray2, -1, kernelV)
            image_convolvedD = cv2.filter2D(image_gray3, -1, kernelD)
            image_convolvedL = cv2.filter2D(image_gray4, -1, kernelL)

            image_gray1 = image_convolvedH
            image_gray2 = image_convolvedV
            image_gray3 = image_convolvedD
            image_gray4 = image_convolvedL

            IH = image_convolvedH[::2, ::2]
            IV = image_convolvedV[::2, ::2]
            ID = image_convolvedD[::2, ::2]
            IL = image_convolvedL[::2, ::2]

            imagenes_juntas = np.hstack((IH,IV,ID,IL))

            cv2.imshow("uu",imagenes_juntas)

            cv2.waitKey(0)

    def infoGris(self,img):
        imgris = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        num_rows, num_cols = (imgris.shape[0], imgris.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        half_size = num_rows / 2
        high_pass_mask = np.zeros_like(imgris)

        return imgris,half_size,high_pass_mask,col_iter,row_iter,num_rows, num_cols





imagen = Cambiotamano(r'C:\Users\sngh9\OneDrive\Escritorio\Maestria_Semestre_2\Procesamiento_de_imagenes\Taller_3\lena.png')
#imagen2 = cv2.imread(r'C:\Users\sngh9\OneDrive\Escritorio\Maestria_Semestre_2\Procesamiento_de_imagenes\Taller_3\lena.png')
imagen.Diezmado(3)
imagen.interpolacion(5)
imagen.descomposicion(3)