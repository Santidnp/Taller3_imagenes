def descomposicion(self, N):
    image = cv2.imread(self.path)
    self.N = N
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

        IH = self.diezmado(2, image_convolvedH)
        IV = self.diezmado(2, image_convolvedV)
        ID = self.diezmado(2, image_convolvedD)
        IL = self.diezmado(2, image_convolvedL)