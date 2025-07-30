# filepath: /object-color-detector-netlify/object-color-detector-netlify/src/object_color_detector.py
import cv2
import numpy as np
from typing import List, Dict
from flask import Flask, request, jsonify

app = Flask(__name__)

class SpecificObjectDetector:
    def __init__(self):
        self.target_colors = {
            'azul': {'lower': np.array([100, 120, 70]), 'upper': np.array([130, 255, 255])},
            'dourado': {'lower': np.array([15, 100, 100]), 'upper': np.array([35, 255, 255])},
            'prata': {'lower': np.array([0, 0, 180]), 'upper': np.array([180, 25, 255])}
        }
        
        self.min_area = 800
        self.max_area = 15000
        self.min_arc_length = 100
        self.max_width_height_ratio = 4.0
        self.min_curvature_score = 0.3
        
    def remove_background_and_skin(self, image: np.ndarray, hsv: np.ndarray) -> np.ndarray:
        skin_lower1 = np.array([0, 20, 70])
        skin_upper1 = np.array([20, 150, 255])
        skin_mask1 = cv2.inRange(hsv, skin_lower1, skin_upper1)
        
        skin_lower2 = np.array([170, 20, 70])
        skin_upper2 = np.array([180, 150, 255])
        skin_mask2 = cv2.inRange(hsv, skin_lower2, skin_upper2)
        
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        
        clean_mask = cv2.bitwise_not(skin_mask)
        return clean_mask
    
    def analyze_object_shape(self, contour: np.ndarray) -> Dict:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area < self.min_area or perimeter == 0:
            return {'is_target_object': False, 'reason': 'área_muito_pequena'}
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        
        if aspect_ratio > self.max_width_height_ratio:
            return {'is_target_object': False, 'reason': 'muito_alongado'}
        
        compactness = 4 * np.pi * area / (perimeter * perimeter)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        
        curvature_score = 0
        
        if 0.1 < compactness < 0.6:
            curvature_score += 0.3
        
        if 0.6 < solidity < 0.95:
            curvature_score += 0.2
        
        if 4 < num_vertices < 15:
            curvature_score += 0.3
        
        perimeter_area_ratio = perimeter / np.sqrt(area) if area > 0 else 0
        if perimeter_area_ratio > 8:
            curvature_score += 0.2
        
        is_target = (curvature_score >= self.min_curvature_score and 
                     area >= self.min_area and 
                     area <= self.max_area and
                     perimeter >= self.min_arc_length)
        
        return {
            'is_target_object': is_target,
            'curvature_score': curvature_score,
            'compactness': compactness,
            'solidity': solidity,
            'num_vertices': num_vertices,
            'aspect_ratio': aspect_ratio,
            'perimeter_area_ratio': perimeter_area_ratio,
            'area': area,
            'perimeter': perimeter
        }
    
    def detect_target_objects(self, image: np.ndarray) -> List[Dict]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_filtered = cv2.bilateralFilter(image, 9, 75, 75)
        hsv_filtered = cv2.cvtColor(image_filtered, cv2.COLOR_BGR2HSV)
        
        clean_mask = self.remove_background_and_skin(image, hsv_filtered)
        
        gray = cv2.cvtColor(image_filtered, cv2.COLOR_BGR2GRAY)
        gray_clean = cv2.bitwise_and(gray, gray, mask=clean_mask)
        
        edges1 = cv2.Canny(gray_clean, 30, 100)
        edges2 = cv2.Canny(gray_clean, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        object_id = 0
        
        for contour in contours:
            shape_analysis = self.analyze_object_shape(contour)
            
            if not shape_analysis['is_target_object']:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            padding = 3
            x_roi = max(0, x - padding)
            y_roi = max(0, y - padding)
            w_roi = min(image.shape[1] - x_roi, w + 2*padding)
            h_roi = min(image.shape[0] - y_roi, h + 2*padding)
            
            roi_hsv = hsv_filtered[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
            detected_color = self.identify_object_color(roi_hsv, contour, x_roi, y_roi)
            
            if detected_color == 'não_identificado':
                continue
            
            detected_objects.append({
                'id': object_id,
                'color': detected_color,
                'contour': contour,
                'bbox': (x, y, w, h),
                'shape_analysis': shape_analysis,
                'center': (x + w//2, y + h//2)
            })
            
            object_id += 1
        
        return detected_objects
    
    def identify_object_color(self, roi_hsv: np.ndarray, contour: np.ndarray, offset_x: int, offset_y: int) -> str:
        best_color = 'não_identificado'
        best_score = 0
        
        for color_name, color_range in self.target_colors.items():
            color_mask = cv2.inRange(roi_hsv, color_range['lower'], color_range['upper'])
            contour_adjusted = contour - [offset_x, offset_y]
            roi_shape = roi_hsv.shape[:2]
            contour_mask = np.zeros(roi_shape, dtype=np.uint8)
            cv2.fillPoly(contour_mask, [contour_adjusted], 255)
            intersection = cv2.bitwise_and(color_mask, contour_mask)
            color_pixels = cv2.countNonZero(intersection)
            total_pixels = cv2.countNonZero(contour_mask)
            
            if total_pixels > 0:
                score = color_pixels / total_pixels
                
                if score > 0.4 and score > best_score:
                    best_score = score
                    best_color = color_name
        
        return best_color if best_score > 0.4 else 'não_identificado'
    
    def draw_results(self, image: np.ndarray, objects: List[Dict]) -> np.ndarray:
        result = image.copy()
        
        contour_colors = {
            'azul': (255, 0, 0),
            'dourado': (0, 215, 255),
            'prata': (192, 192, 192)
        }
        
        status_messages = {
            'azul': {'text': 'CAVACO QUEIMADO', 'color': (0, 0, 255)},
            'dourado': {'text': 'ATENCAO', 'color': (0, 165, 255)},
            'prata': {'text': 'OK', 'color': (0, 255, 0)}
        }
        
        for obj in objects:
            color_name = obj['color']
            x, y, w, h = obj['bbox']
            contour_color = contour_colors[color_name]
            cv2.drawContours(result, [obj['contour']], -1, contour_color, 3)
            status_info = status_messages[color_name]
            status_text = status_info['text']
            status_color = status_info['color']
            main_label = "CAVACO ESPIRALADO"
            (main_w, main_h), _ = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(result, (x, y - main_h - 35), (x + main_w + 10, y - 30), (50, 50, 50), -1)
            cv2.putText(result, main_label, (x + 5, y - 32), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            (status_w, status_h), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
            cv2.rectangle(result, (x, y - 25), (x + status_w + 10, y + 5), (0, 0, 0), -1)
            cv2.putText(result, status_text, (x + 5, y - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 3)
            cv2.circle(result, obj['center'], 5, contour_color, -1)
        
        if len(objects) > 0:
            legend_y = 30
            cv2.rectangle(result, (10, 5), (500, 80), (0, 0, 0), -1)
            cv2.putText(result, "DETECAO DE CAVACOS ESPIRALADOS:", (15, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result, "AZUL=QUEIMADO | DOURADO=ATENCAO | PRATA=OK", (15, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(result, f"TOTAL DETECTADOS: {len(objects)}", (15, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result
    
    def analyze_image(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erro: Não foi possível carregar {image_path}")
            return [], None
        
        print(f"Analisando: {image_path}")
        print("Procurando cavacos espiralados...")
        
        objects = self.detect_target_objects(image)
        
        print(f"\n=== CAVACOS ESPIRALADOS DETECTADOS ===")
        if len(objects) == 0:
            print("❌ Nenhum cavaco espiralado encontrado")
        else:
            print(f"✅ {len(objects)} cavaco(s) espiralado(s) encontrado(s):")
            for i, obj in enumerate(objects):
                color = obj['color']
                if color == 'azul':
                    status = "CAVACO QUEIMADO ⚠️"
                elif color == 'dourado':
                    status = "ATENÇÃO ⚠️"
                else:
                    status = "OK ✅"
                
                print(f"   {i+1}. {color.upper()} → {status}")
        
        result = self.draw_results(image, objects)
        return objects, result
    
    def show_detection_details(self, image_path: str):
        objects, result = self.analyze_image(image_path)
        
        return objects
    
    def get_led_status(self, objects: list) -> dict:
        """
        Retorna qual LED deve ser aceso baseado na prioridade:
        Azul > Dourado > Prata.
        Exemplo de saída:
        {
            "led_azul": True/False,
            "led_dourado": True/False,
            "led_prata": True/False,
            "cor_detectada": "azul"/"dourado"/"prata"/"nenhum"
        }
        """
        status = {"led_azul": False, "led_dourado": False, "led_prata": False, "cor_detectada": "nenhum"}
        cores = [obj['color'] for obj in objects]
        if 'azul' in cores:
            status["led_azul"] = True
            status["cor_detectada"] = "azul"
        elif 'dourado' in cores:
            status["led_dourado"] = True
            status["cor_detectada"] = "dourado"
        elif 'prata' in cores:
            status["led_prata"] = True
            status["cor_detectada"] = "prata"
        return status

detector = SpecificObjectDetector()

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    if not file:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400
    
    # Ler a imagem como um array numpy
    in_memory_file = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({"error": "Não foi possível decodificar a imagem"}), 400
    
    # Detectar objetos na imagem
    objects = detector.detect_target_objects(image)
    led_status = detector.get_led_status(objects)
    return jsonify({"objetos": objects, "led_status": led_status})