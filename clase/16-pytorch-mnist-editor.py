import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy  as np
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Tuple
from abc import ABC, abstractmethod

MNIST_DATASET = None
NORMALIZED_TRAIN_LOADER = None
NORMALIZED_VAL_LOADER = None

def load_and_normalize_mnist(message_queue=None):
    global MNIST_DATASET, NORMALIZED_TRAIN_LOADER, NORMALIZED_VAL_LOADER
    
    def update_message(msg):
        if message_queue:
            message_queue.put(msg)
        else:
            print(msg)
    
    if MNIST_DATASET is None:
        update_message("Cargando dataset MNIST...")
        MNIST_DATASET = datasets.MNIST(
            root='./data', 
            train=True, 
            transform=transforms.ToTensor(),
            download=True
        )
        
        total_images = len(MNIST_DATASET)
        normalized_images = []
        last_percentage = -1
        
        update_message("Normalizando imágenes...")
        for i, (img, _) in enumerate(MNIST_DATASET):
            current_percentage = int((i / total_images) * 100)
            if current_percentage > last_percentage:
                update_message(f"Normalizando imágenes: {current_percentage}%")
                last_percentage = current_percentage
            normalized_images.append(DigitRecognizer.normalize_image(img.squeeze()))
        normalized_images = torch.stack(normalized_images)
        
        update_message("Preparando datasets...")
        # Crear nuevo dataset con imágenes normalizadas
        normalized_dataset = [
            (img.view(-1), label) 
            for img, (_, label) in zip(normalized_images, MNIST_DATASET)
        ]
        
        # Dividir en train y validación
        train_size = int(0.8 * len(normalized_dataset))
        val_size = len(normalized_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            normalized_dataset, [train_size, val_size]
        )
        
        update_message("Creando data loaders...")
        # Crear dataloaders
        NORMALIZED_TRAIN_LOADER = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True
        )
        NORMALIZED_VAL_LOADER = torch.utils.data.DataLoader(
            val_dataset, batch_size=64, shuffle=False
        )
        update_message("Dataset preparado y normalizado!")

# Configuración
@dataclass
class Config:
    PIXEL_SIZE: int = 20
    GRID_SIZE: int = 28
    MARGIN: int = 20
    
    @property
    def WINDOW_SIZE(self) -> int:
        return self.PIXEL_SIZE * self.GRID_SIZE
    
    @property
    def TOTAL_WINDOW_WIDTH(self) -> int:
        return self.WINDOW_SIZE * 2 + self.MARGIN * 2
    
    @property
    def TOTAL_WINDOW_HEIGHT(self) -> int:
        return self.WINDOW_SIZE + 120 + self.MARGIN * 2

class Colors:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (200, 200, 200)
    LIGHT_GRAY = (230, 230, 230)
    DARK_GRAY = (100, 100, 100)
    LIGHT_BLUE = (173, 216, 230)
    VERY_LIGHT_GRAY = (245, 245, 245)
    BUTTON_PRESSED = (150, 190, 210)
    AUTOMATIC_MODE = (180, 180, 180)

class DigitRecognizer(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None  # Será definido en las clases hijas
    
    @abstractmethod
    def setup_architecture(self):
        """Define la arquitectura de la red neuronal"""
        pass
    
    @abstractmethod
    def forward(self, x):
        """Implementa el paso forward de la red"""
        pass

    @staticmethod
    def normalize_image(image):
        """Normaliza una imagen centrándola en la grilla."""
        # Encontrar píxeles no vacíos
        non_zero = torch.where(image > 0)
        if len(non_zero[0]) == 0:
            return image
        
        # Obtener límites
        min_y, max_y = non_zero[0].min(), non_zero[0].max()
        min_x, max_x = non_zero[1].min(), non_zero[1].max()
        
        # Extraer el dibujo
        drawing = image[min_y:max_y+1, min_x:max_x+1]
        
        # Crear nueva imagen vacía
        new_image = torch.zeros_like(image)
        
        # Calcular posición central
        height, width = drawing.shape
        start_y = (28 - height) // 2
        start_x = (28 - width) // 2
        
        # Colocar el dibujo centrado
        new_image[start_y:start_y+height, start_x:start_x+width] = drawing
        
        return new_image

    def score(self, test_loader):
        """Evalúa la precisión del modelo en el conjunto de datos proporcionado"""
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self(images)  # Las imágenes ya vienen normalizadas
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def train_model(self, train_loader, val_loader, progress_callback=None):
        """Entrena el modelo y evalúa su precisión"""
        total_batches = len(train_loader)
        self.train()
        
        for _ in range(1):  # Una época
            for batch_idx, (images, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()
                outputs = self(images)  # Las imágenes ya vienen normalizadas
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                if progress_callback:
                    # Calcular porcentaje sin decimales
                    progress = int((batch_idx + 1) / total_batches * 100)
                    progress_callback(progress)
        
        return self.score(val_loader)

class DigitSimple(DigitRecognizer):
    def __init__(self):
        super().__init__()
        self.setup_architecture()
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)
    
    def setup_architecture(self):
        self.fc1 = nn.Linear(784, 10)  # Capa única que conecta entrada con salida
    
    def forward(self, x):
        return self.fc1(x)  # Forward pass directo sin función de activación

class DigitMultiple(DigitRecognizer):
    def __init__(self):
        super().__init__()
        self.setup_architecture()
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)
    
    def setup_architecture(self):
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DigitConvolucion(DigitRecognizer):
    def __init__(self):
        super().__init__()
        self.setup_architecture()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    
    def setup_architecture(self):
        # Capas convolucionales
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_dropout = nn.Dropout2d()  # Para capas convolucionales
        self.fc_dropout = nn.Dropout()      # Para capas fully connected
        # Capas fully connected
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        # Reshape para la entrada convolucional si viene aplanada
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        
        # Capas convolucionales
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        x = self.conv_dropout(x)  # Dropout2d para capas convolucionales
        x = torch.relu(torch.max_pool2d(x, 2))
        
        # Aplanar para las capas fully connected
        x = x.view(-1, 320)
        # Capas fully connected
        x = torch.relu(self.fc1(x))
        x = self.fc_dropout(x)    # Dropout normal para capas fully connected
        x = self.fc2(x)
        return x

class Button:
    def __init__(self, x: int, y: int, width: int, height: int, text: str, icon: str = None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.icon = icon
        self.pressed = False
        self.active = True
        self.momentary_pressed = False

    def draw(self, surface: pygame.Surface):
        # Dibujar sombra
        shadow_offset = 3
        if not self.momentary_pressed:
            pygame.draw.rect(surface, Colors.DARK_GRAY,
                           (self.rect.x + shadow_offset, 
                            self.rect.y + shadow_offset,
                            self.rect.width, self.rect.height),
                           border_radius=10)

        # Dibujar botón
        color = (Colors.BUTTON_PRESSED if self.momentary_pressed else 
                Colors.LIGHT_BLUE if self.active else 
                Colors.VERY_LIGHT_GRAY)
        offset = 2 if self.momentary_pressed else 0
        
        pygame.draw.rect(surface, color,
                        (self.rect.x + offset,
                         self.rect.y + offset,
                         self.rect.width,
                         self.rect.height),
                        border_radius=10)
        
        if self.icon:
            # Dibujar ícono del tacho
            trash_color = Colors.BLACK if self.active else Colors.LIGHT_GRAY
            x = self.rect.centerx + offset
            y = self.rect.centery + offset
            
            # Base del tacho
            pygame.draw.rect(surface, trash_color, 
                           (x-8, y-2, 16, 12), 2)
            # Tapa del tacho
            pygame.draw.rect(surface, trash_color, 
                           (x-10, y-6, 20, 4))
            # Mango de la tapa
            pygame.draw.rect(surface, trash_color, 
                           (x-4, y-8, 8, 2))
            # Líneas verticales
            for i in range(-6, 7, 6):
                pygame.draw.line(surface, trash_color,
                               (x+i, y-1), (x+i, y+8))
        else:
            # Dibujar texto con el mismo offset que el botón
            font = pygame.font.SysFont(None, 24)
            text_surf = font.render(self.text, True,
                                  Colors.BLACK if self.active else Colors.LIGHT_GRAY)
            text_rect = text_surf.get_rect(center=(
                self.rect.centerx + offset,
                self.rect.centery + offset
            ))
            surface.blit(text_surf, text_rect)

class DrawingGrid:
    def __init__(self, config: Config):
        self.config = config
        # Cambiar a float32 para manejar intensidades
        self.grid = np.zeros((config.GRID_SIZE, config.GRID_SIZE), dtype=np.float32)
        self.auto_clear = False
        self.edit_mode = "wait"
        self.last_pos = None  # Añadir tracking de última posición

    def draw(self, surface: pygame.Surface):
        for y in range(self.config.GRID_SIZE):
            for x in range(self.config.GRID_SIZE):
                rect = pygame.Rect(
                    self.config.MARGIN + x * self.config.PIXEL_SIZE,
                    self.config.MARGIN * 3 + y * self.config.PIXEL_SIZE,
                    self.config.PIXEL_SIZE,
                    self.config.PIXEL_SIZE
                )
                # Calcular color basado en la intensidad
                intensity = int(self.grid[y, x] * 255)
                if self.auto_clear:
                    color = Colors.AUTOMATIC_MODE if intensity > 0 else Colors.WHITE
                else:
                    color = (255 - intensity, 255 - intensity, 255 - intensity)
                
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, Colors.GRAY, rect, 1)

    def paint_pixel_and_surroundings(self, x: int, y: int):
        if 0 <= x < self.config.GRID_SIZE and 0 <= y < self.config.GRID_SIZE:
            # Pintar pixel central con intensidad 1.0
            self.grid[y, x] = 1.0
            
            # Pintar los 8 pixeles alrededor con intensidad 0.5
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue  # Saltar el pixel central
                    new_x, new_y = x + dx, y + dy
                    if (0 <= new_x < self.config.GRID_SIZE and 
                        0 <= new_y < self.config.GRID_SIZE):
                        self.grid[new_y, new_x] = max(0.5, self.grid[new_y, new_x])

    def clear(self):
        self.grid.fill(0.0)
        self.auto_clear = False

    def interpolate_points(self, x1, y1, x2, y2):
        """Interpola puntos entre dos coordenadas."""
        points = []
        dx = x2 - x1
        dy = y2 - y1
        steps = max(abs(dx), abs(dy)) + 1
        
        if steps > 1:
            x_step = dx / steps
            y_step = dy / steps
            
            for i in range(steps):
                x = int(x1 + x_step * i)
                y = int(y1 + y_step * i)
                if 0 <= x < self.config.GRID_SIZE and 0 <= y < self.config.GRID_SIZE:
                    points.append((x, y))
        
        return points

    def get_drawing_bounds(self):
        """Encuentra los límites del dibujo en la grilla."""
        # Encontrar píxeles no vacíos
        non_zero = np.where(self.grid > 0)
        if len(non_zero[0]) == 0:  # Si no hay dibujo
            return None
        
        # Obtener límites
        min_y, max_y = non_zero[0].min(), non_zero[0].max()
        min_x, max_x = non_zero[1].min(), non_zero[1].max()
        
        return min_x, max_x, min_y, max_y
    
    def normalize_drawing(self):
        """Centra el dibujo sin cambiar su tamaño."""
        bounds = self.get_drawing_bounds()
        if not bounds:
            return
        
        min_x, max_x, min_y, max_y = bounds
        
        # Calcular dimensiones actuales
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        
        # Extraer el dibujo original
        drawing = self.grid[min_y:max_y+1, min_x:max_x+1].copy()
        
        # Crear nueva grilla vacía
        new_grid = np.zeros_like(self.grid)
        
        # Calcular posición central
        start_y = (self.config.GRID_SIZE - height) // 2
        start_x = (self.config.GRID_SIZE - width) // 2
        
        # Colocar el dibujo centrado
        new_grid[start_y:start_y+height, start_x:start_x+width] = drawing
        
        self.grid = new_grid

class DigitRecognizerApp:
    def __init__(self):
        pygame.init()
        self.config = Config()
        self.setup_window()
        self.setup_components()
        self.setup_model()
        self.pressed_button = None  # Nuevo: para tracking del botón presionado
        self.mnist_dataset = None  # Añadir referencia al dataset
        self.current_model = 'simple'  # Para trackear qué modelo está activo
        
    def setup_window(self):
        self.window = pygame.display.set_mode(
            (self.config.TOTAL_WINDOW_WIDTH, self.config.TOTAL_WINDOW_HEIGHT)
        )
        pygame.display.set_caption("Demo de reconocimiento de dígitos")

    def setup_components(self):
        self.grid = DrawingGrid(self.config)
        self.buttons = self.create_buttons()
        self.message_queue = queue.Queue()
        self.info_message = ""
        self.recognized_digit = None
        self.disable_buttons = False
        self.buttons['recognize'].active = False  # Desactivar inicialmente
        self.current_label = None

    def setup_model(self):
        self.model_simple = DigitSimple()
        self.model_multiple = DigitMultiple()
        self.model_conv = DigitConvolucion()
        self.model = self.model_simple  # Modelo activo por defecto
        self.model_trained = False

    def create_buttons(self) -> dict:
        train_width = 110  # Botones de entrenamiento m��s anchos
        button_width = 90
        button_height = 40
        button_y = self.config.WINDOW_SIZE + self.config.MARGIN * 3 + 20
        spacing = 10
        
        # Posición del botón borrar con margen de 5 píxeles desde el borde de la grilla
        clear_x = self.config.MARGIN + self.config.WINDOW_SIZE - button_height - 5  # 5 píxeles desde el borde derecho
        clear_y = self.config.MARGIN * 3 + self.config.WINDOW_SIZE - button_height - 5  # 5 píxeles desde el borde inferior
        
        return {
            'train_simple': Button(self.config.MARGIN, button_y, train_width, button_height, 'Simple'),
            'train_multiple': Button(self.config.MARGIN + train_width + spacing, button_y, train_width, button_height, 'Multiple'),
            'train_conv': Button(self.config.MARGIN + (train_width + spacing) * 2, button_y, train_width, button_height, 'Convolución'),
            'recognize': Button(self.config.MARGIN + (train_width + spacing) * 3, button_y, button_width, button_height, 'Reconocer'),
            'example': Button(self.config.MARGIN + (train_width + spacing) * 3 + button_width + spacing, button_y, button_width, button_height, 'Ejemplo'),
            'clear': Button(clear_x, clear_y, button_height, button_height, '', icon='trash'),
            'exit': Button(
                self.config.TOTAL_WINDOW_WIDTH - button_width - self.config.MARGIN,
                self.config.TOTAL_WINDOW_HEIGHT - button_height - self.config.MARGIN,
                button_width, 
                button_height, 
                'Salir'
            )
        }

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            pygame.display.flip()
        pygame.quit()

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN and not self.disable_buttons:
                if event.key == pygame.K_e:  # Tecla 'E' para Entrenar
                    self.handle_button_action('train')
                elif event.key == pygame.K_r:  # Tecla 'R' para Reconocer
                    self.handle_button_action('recognize')
                elif event.key == pygame.K_b:  # Tecla 'B' para Borrar
                    self.handle_button_action('clear')
                elif event.key == pygame.K_s:  # Tecla 'S' para Salir
                    self.handle_button_action('exit')
                
            if event.type == pygame.MOUSEBUTTONDOWN and not self.disable_buttons:
                mouse_pos = pygame.mouse.get_pos()
                for name, button in self.buttons.items():
                    if button.rect.collidepoint(mouse_pos):
                        button.momentary_pressed = True
                        self.pressed_button = name
                        
            elif event.type == pygame.MOUSEBUTTONUP:
                self.grid.last_pos = None
                if self.pressed_button and not self.disable_buttons:
                    self.handle_button_action(self.pressed_button)
                self.pressed_button = None
                for button in self.buttons.values():
                    button.momentary_pressed = False

            # Manejo del dibujo en la cuadrícula
            mouse_buttons = pygame.mouse.get_pressed()
            if mouse_buttons[0]:
                self.handle_drawing(pygame.mouse.get_pos())
            else:
                self.grid.edit_mode = "wait"

        return True

    def handle_button_action(self, action: str):
        if action == 'train_simple':
            self.current_model = 'simple'
            self.model = self.model_simple
            self.disable_buttons = True
            for button in self.buttons.values():
                button.active = False
            self.start_training()
        elif action == 'train_multiple':
            self.current_model = 'multiple'
            self.model = self.model_multiple
            self.disable_buttons = True
            for button in self.buttons.values():
                button.active = False
            self.start_training()
        elif action == 'train_conv':
            self.current_model = 'conv'
            self.model = self.model_conv
            self.disable_buttons = True
            for button in self.buttons.values():
                button.active = False
            self.start_training()
        elif action == 'recognize' and self.model_trained:
            self.recognize()
        elif action == 'example':
            self.load_random_example()
        elif action == 'clear':
            self.grid.clear()
            self.recognized_digit = None
            self.current_label = None
        elif action == 'exit':
            pygame.quit()
            exit()  # Salir completamente del programa

    def handle_drawing(self, pos):
        x, y = pos
        margin3 = self.config.MARGIN * 3
        if margin3 <= y < margin3 + self.config.WINDOW_SIZE:
            if self.grid.auto_clear:
                self.grid.clear()
                self.grid.auto_clear = False

            grid_x = (x - self.config.MARGIN) // self.config.PIXEL_SIZE
            grid_y = (y - margin3) // self.config.PIXEL_SIZE
            
            if 0 <= grid_x < self.config.GRID_SIZE and 0 <= grid_y < self.config.GRID_SIZE:
                if self.grid.edit_mode == "wait":
                    self.grid.edit_mode = "paint" if self.grid.grid[grid_y, grid_x] < 0.5 else "erase"
                
                if self.grid.last_pos is None:
                    self.grid.last_pos = (grid_x, grid_y)
                
                if self.grid.edit_mode == "paint":
                    # Interpolar entre la última posición y la actual
                    points = self.grid.interpolate_points(
                        self.grid.last_pos[0], self.grid.last_pos[1],
                        grid_x, grid_y
                    )
                    for px, py in points:
                        self.grid.paint_pixel_and_surroundings(px, py)
                elif self.grid.edit_mode == "erase":
                    points = self.grid.interpolate_points(
                        self.grid.last_pos[0], self.grid.last_pos[1],
                        grid_x, grid_y
                    )
                    for px, py in points:
                        self.grid.grid[py, px] = 0.0
                
                self.grid.last_pos = (grid_x, grid_y)
        else:
            self.grid.last_pos = None

    def update(self):
        while not self.message_queue.empty():
            message = self.message_queue.get()
            if message == "HABILITAR_BOTONES":
                self.disable_buttons = False
                for name, button in self.buttons.items():
                    # Solo habilitar 'recognize' si el modelo está entrenado
                    if name == 'recognize':
                        button.active = self.model_trained
                    else:
                        button.active = True
            else:
                self.info_message = message

    def draw(self):
        self.window.fill(Colors.WHITE)
        
        # Dibujar título
        font = pygame.font.SysFont(None, 48)
        title = font.render('Demo de reconocimiento de dígitos', True, Colors.BLACK)
        title_rect = title.get_rect(center=(self.config.TOTAL_WINDOW_WIDTH // 2, self.config.MARGIN))
        self.window.blit(title, title_rect)
        
        # Dibujar cuadrícula
        self.grid.draw(self.window)
        
        # Dibujar botones
        for button in self.buttons.values():
            button.draw(self.window)
        
        # Dibujar mensaje de información (10 píxeles más abajo)
        info_font = pygame.font.SysFont(None, 24)
        info_surf = info_font.render(self.info_message, True, Colors.BLACK)
        self.window.blit(info_surf, (self.config.MARGIN, 
                                   self.config.WINDOW_SIZE + self.config.MARGIN * 4 + 50))  # Cambiado de 40 a 50
        
        # Dibujar dígito reconocido y probabilidad
        if self.recognized_digit is not None and hasattr(self, 'recognition_prob'):
            # Dibujar el dígito
            font = pygame.font.SysFont(None, 480)
            digit_surf = font.render(str(self.recognized_digit), True, Colors.BLACK)
            digit_rect = digit_surf.get_rect(center=(
                self.config.TOTAL_WINDOW_WIDTH - self.config.WINDOW_SIZE // 2 - self.config.MARGIN,
                self.config.TOTAL_WINDOW_HEIGHT // 2
            ))
            self.window.blit(digit_surf, digit_rect)
            
            # Dibujar la probabilidad
            prob_font = pygame.font.SysFont(None, 36)
            prob_text = f"Probabilidad: {self.recognition_prob:.1%}"
            prob_surf = prob_font.render(prob_text, True, Colors.BLACK)
            prob_rect = prob_surf.get_rect(center=(
                digit_rect.centerx,
                digit_rect.bottom + 30
            ))
            self.window.blit(prob_surf, prob_rect)

    def start_training(self):
        self.disable_buttons = True
        threading.Thread(target=self.train_model, daemon=True).start()

    def train_model(self):
        try:
            model_type = {
                'simple': 'simple',
                'multiple': 'múltiple',
                'conv': 'convolucional'
            }[self.current_model]
            
            # Si es necesario cargar los datos
            if MNIST_DATASET is None:
                load_and_normalize_mnist(self.message_queue)
        
            self.message_queue.put(f"Iniciando entrenamiento red {model_type}...")
        
            def update_progress(progress):
                self.message_queue.put(f"Entrenando red {model_type}: {progress}%")
            
            accuracy = self.model.train_model(
                NORMALIZED_TRAIN_LOADER, 
                NORMALIZED_VAL_LOADER, 
                update_progress
            )
            
            self.model_trained = True
            self.buttons['recognize'].active = True
            self.message_queue.put(f"Entrenamiento {model_type} completado - Precisión: {accuracy:.1%}")
        except Exception as e:
            self.message_queue.put(f"Error en el entrenamiento: {str(e)}")
        finally:
            self.message_queue.put("HABILITAR_BOTONES")

    def load_random_example(self):
        if MNIST_DATASET is None:
            self.info_message = "Primero debe entrenar el modelo"
            return
        
        # Desactivar auto_clear para permitir escalas de grises
        self.grid.auto_clear = False
        
        # Seleccionar ejemplo aleatorio
        idx = np.random.randint(0, len(MNIST_DATASET))
        image, label = MNIST_DATASET[idx]
        
        # Asegurarnos de preservar las intensidades
        image_np = image.squeeze().numpy()
        normalized_image = self.model.normalize_image(torch.tensor(image_np))
        
        # Asegurarnos de que los valores estén entre 0 y 1
        self.grid.grid = normalized_image.numpy().clip(0, 1)
        
        self.current_label = label.item() if torch.is_tensor(label) else label
        self.info_message = f"Ejemplo cargado - Dígito real: {self.current_label}"

    def recognize(self):
        if not self.model_trained:
            self.info_message = "Primero debe entrenar el modelo"
            return
        
        # Normalizar el dibujo antes de reconocerlo
        self.grid.normalize_drawing()
            
        input_tensor = torch.tensor(self.grid.grid, dtype=torch.float32).view(-1, 28*28)
        with torch.no_grad():
            self.model.eval()
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            self.recognized_digit = torch.argmax(output, dim=1).item()
            self.recognition_prob = probabilities[0, self.recognized_digit].item()
        
        self.info_message = f"Dígito reconocido: {self.recognized_digit} (Confianza: {self.recognition_prob:.1%})"
        
        # Activar el borrado automático para el próximo dibujo
        self.grid.auto_clear = True

if __name__ == "__main__":
    app = DigitRecognizerApp()
    app.run()
