import cv2
import numpy as np


class ImagePreprocessor:
    """
    Tiền xử lý ảnh để cải thiện chất lượng cho OCR
    """
    
    @staticmethod
    def enhance_contrast(image):
        """Tăng cường độ tương phản bằng CLAHE"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced
    
    @staticmethod
    def denoise(image):
        """Giảm nhiễu"""
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        return denoised
    
    @staticmethod
    def sharpen(image):
        """Làm sắc nét ảnh"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
    
    @staticmethod
    def binarize_adaptive(image):
        """Nhị phân hóa thích nghi (tốt cho ảnh có độ sáng không đều)"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        return binary
    
    @staticmethod
    def binarize_otsu(image):
        """Nhị phân hóa Otsu (tốt cho ảnh có phân bố 2 đỉnh rõ ràng)"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    @staticmethod
    def extract_channel(image, channel='green'):
        """Trích xuất kênh màu cụ thể (hữu ích khi văn bản có màu đặc trưng)"""
        if len(image.shape) != 3:
            return image
        
        if channel == 'green':
            return image[:, :, 1]
        elif channel == 'red':
            return image[:, :, 2]
        elif channel == 'blue':
            return image[:, :, 0]
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def morphology_operation(image, operation='close', kernel_size=3):
        """Các phép toán hình thái học để làm sạch ảnh nhị phân"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        if operation == 'close':
            result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operation == 'open':
            result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'dilate':
            result = cv2.dilate(image, kernel, iterations=1)
        elif operation == 'erode':
            result = cv2.erode(image, kernel, iterations=1)
        else:
            result = image
        
        return result
    
    @staticmethod
    def preprocess_pipeline_v1(image):
        """
        Pipeline cơ bản: Denoise -> Enhance -> Sharpen -> Binarize
        Tốt cho ảnh có độ tương phản thấp
        """
        # Bước 1: Giảm nhiễu
        denoised = ImagePreprocessor.denoise(image)
        
        # Bước 2: Tăng cường độ tương phản
        enhanced = ImagePreprocessor.enhance_contrast(denoised)
        
        # Bước 3: Làm sắc nét
        sharpened = ImagePreprocessor.sharpen(enhanced)
        
        # Bước 4: Nhị phân hóa
        binary = ImagePreprocessor.binarize_adaptive(sharpened)
        
        return binary
    
    @staticmethod
    def preprocess_pipeline_v2(image):
        """
        Pipeline nâng cao: Extract channel -> Enhance -> Sharpen -> Otsu
        Tốt cho ảnh có văn bản màu xanh lá/vàng
        """
        # Bước 1: Trích xuất kênh xanh lá (văn bản màu xanh/vàng thường nổi bật ở kênh này)
        green_channel = ImagePreprocessor.extract_channel(image, 'green')
        
        # Bước 2: Tăng cường độ tương phản
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(green_channel)
        
        # Bước 3: Làm sắc nét mạnh
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Bước 4: Nhị phân hóa Otsu
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Bước 5: Làm sạch bằng morphology
        binary = ImagePreprocessor.morphology_operation(binary, 'close', 2)
        
        return binary
    
    @staticmethod
    def preprocess_pipeline_v3(image):
        """
        Pipeline cho ảnh mờ và độ tương phản thấp
        Unsharp masking + Multiple threshold methods
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Unsharp masking: làm sắc nét bằng cách trừ ảnh mờ
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        
        # CLAHE mạnh
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(sharpened)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 5
        )
        
        return binary
    
    @staticmethod
    def compare_methods(image):
        """
        So sánh các phương pháp tiền xử lý khác nhau
        Trả về dict của các kết quả
        """
        results = {
            'original': image,
            'pipeline_v1': ImagePreprocessor.preprocess_pipeline_v1(image),
            'pipeline_v2': ImagePreprocessor.preprocess_pipeline_v2(image),
            'pipeline_v3': ImagePreprocessor.preprocess_pipeline_v3(image),
            'simple_otsu': ImagePreprocessor.binarize_otsu(image),
            'simple_adaptive': ImagePreprocessor.binarize_adaptive(image),
        }
        return results


if __name__ == '__main__':
    # Test với ảnh mẫu
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = '../images/1.jpg'
    
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        sys.exit(1)
    
    print("🔍 So sánh các phương pháp tiền xử lý...")
    results = ImagePreprocessor.compare_methods(image)
    
    # Hiển thị kết quả
    for name, img in results.items():
        cv2.imshow(name, img)
    
    print("\n📊 Các phương pháp:")
    print("  - original: Ảnh gốc")
    print("  - pipeline_v1: Denoise -> Enhance -> Sharpen -> Adaptive")
    print("  - pipeline_v2: Green channel -> CLAHE -> Sharpen -> Otsu (TỐT CHO ẢNH CỦA BẠN)")
    print("  - pipeline_v3: Unsharp masking -> CLAHE -> Adaptive")
    print("  - simple_otsu: Chỉ Otsu threshold")
    print("  - simple_adaptive: Chỉ Adaptive threshold")
    print("\n💡 Nhấn phím bất kỳ để đóng...")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
