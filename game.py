import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_w, K_s, K_a, K_d
import random 
import numpy as np

WIDTH = 24
HEIGHT = 24
TILE_SIZE = 24
gameOver = False

## [=][=][=][=] AI [=][=][=][=]
n_states = WIDTH*HEIGHT      
n_actions = 4          
goal_state = WIDTH*HEIGHT - 1

learning_rate = 0.8
discount_factor = 0.95
exploration_prob = 0.2
epochs = 1000

Q_table = np.zeros((n_states, n_actions))

## [=][=][=][=] PYGA [=][=][=][=] 
pygame.init()
screen = pygame.display.set_mode((24 * WIDTH, 24 * HEIGHT))
pygame.display.set_caption("Snakers")
clock = pygame.time.Clock()


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class SnakePart:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, surface):
        pygame.draw.rect(surface, (0, 255, 0), (self.x, self.y, 20, 20))
    
    def move(self, dx, dy):
        self.x += dx
        self.y += dy

class Snake:
    def __init__(self):
        self.parts = [SnakePart(50,60)]
        self.direction = (50, 50)
        self.DirY = 0
        self.DirX = 1

        for x in range(2):
            self.add_part()
    
    def add_part(self):
        self.parts.append(SnakePart(self.parts[-1].x + (21 * self.DirX), self.parts[-1].y + (21 * self.DirY)))
        
    def draw(self, surface):
        for part in self.parts:
            part.draw(surface)

    def update(self):
        snake.draw(screen)    

        self.add_part()
        self.parts.pop(0)

    def up(self):
        if not self.DirY == 1:
            self.DirY = -1
            self.DirX = 0
    def down(self):
        if not self.DirY == -1:
            self.DirY = 1
            self.DirX = 0
    def left(self):
        if not self.DirX == 1:
            self.DirX = -1
            self.DirY = 0
    def right(self):
        if not self.DirX == -1:
            self.DirX = 1
            self.DirY = 0
    
class Apple():
    def __init__(self):
        self.randomPos()

    def randomPos(self):
        self.x, self.y = random.randint(0,WIDTH**2 - 24), random.randint(0,HEIGHT**2 - 24)
        
    def draw(self, surface):
        pygame.draw.rect(surface, (255, 0, 0), (self.x, self.y, 20, 20))
        
    def apple_eaten(self):
        self.randomPos()
        snake.add_part() 
        
def is_collision(sprite1, sprite2):
  
  x1, y1 = sprite1.x, sprite1.y
  x2, y2 = sprite2.x, sprite2.y
 
  distance = ((x2 - x1)**2 + (y2 -y1)**2)**0.5
  if(distance < 21):
      return True
  
  return False

snake = Snake()
apple = Apple()

## AI SHIT
apple_grid_x = apple.x // TILE_SIZE
apple_grid_y = apple.y // TILE_SIZE
goal_state = apple_grid_y * WIDTH + apple_grid_x

def get_next_state(state, action):
    row, col = divmod(state, WIDTH)
    
    if action == 0:     
        row -= 1
        snake.up()
    elif action == 1:  
        row += 1
        snake.down()
    elif action == 2:  
        col -= 1
        snake.left()
    elif action == 3:   
        col += 1
        snake.right()

    row = max(0, min(HEIGHT - 1, row))
    col = max(0, min(WIDTH - 1, col))

    
    return row * WIDTH + col


head = snake.parts[-1]
snake_grid_x = head.x // TILE_SIZE
snake_grid_y = head.y // TILE_SIZE
current_state = snake_grid_y * WIDTH + snake_grid_x
gameOver = False

while not gameOver:
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            pygame.quit()
            exit()
        if (event.type == KEYDOWN and event.key == K_w):
            snake.up()

        elif (event.type == KEYDOWN and event.key == K_s):
            snake.down()

        elif (event.type == KEYDOWN and event.key == K_a):
            snake.left()

        elif (event.type == KEYDOWN and event.key == K_d):
            snake.right()


    screen.fill((0, 0, 0))
    apple.draw(screen)
    snake.update()
    if(is_collision(snake.parts[-1], apple)):
        apple.apple_eaten()
        apple_grid_x = apple.x // TILE_SIZE
        apple_grid_y = apple.y // TILE_SIZE
        goal_state = apple_grid_y * WIDTH + apple_grid_x


    for part in snake.parts[:-1]:
        if is_collision(snake.parts[-1], part):
            gameOver = True

    if snake.parts[-1].x >= WIDTH**2 or snake.parts[-1].x <=0:
        gameOver = True
    if snake.parts[-1].y <= 0 or snake.parts[-1].y >= HEIGHT**2:
        gameOver = True
    
    pygame.display.flip()
    clock.tick(7)

    if np.random.rand() < exploration_prob:
        action = np.random.randint(0, n_actions)
    else:
        action = np.argmax(Q_table[current_state])

    next_state = get_next_state(current_state, action)

    Q_table[current_state, action] += learning_rate * (
        reward + discount_factor * np.max(Q_table[next_state]) - Q_table[current_state, action]
    )

    if next_state == goal_state:
        break
    
    current_state = next_state

    print("state:", current_state, "action:", action, "next:", next_state)

