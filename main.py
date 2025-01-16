import pygame

from bots import PolicyBot, HumanBot
from env import CoinCollectionEnv
from rl_bot import RLBot


def handle_human_input():
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        return 0
    elif keys[pygame.K_RIGHT]:
        return 1
    elif keys[pygame.K_DOWN]:
        return 2
    elif keys[pygame.K_LEFT]:
        return 3
    return None

def draw_pause_menu(window):
    # Create semi-transparent overlay
    overlay = pygame.Surface((window.get_width(), window.get_height()))
    overlay.fill((0, 0, 0))
    overlay.set_alpha(180)
    window.blit(overlay, (0, 0))
    
    # Setup font
    font = pygame.font.Font(None, 74)
    
    # Create buttons
    button_width = 200
    button_height = 50
    buttons = {}
    
    # "Continue" button
    continue_rect = pygame.Rect(0, 0, button_width, button_height)
    continue_rect.center = (window.get_width()//2, window.get_height()//2 - 40)
    buttons['continue'] = continue_rect
    
    # "Leave" button
    leave_rect = pygame.Rect(0, 0, button_width, button_height)
    leave_rect.center = (window.get_width()//2, window.get_height()//2 + 40)
    buttons['leave'] = leave_rect
    
    # Draw buttons and text
    for text, rect in buttons.items():
        # Draw button
        pygame.draw.rect(window, (0, 200, 0), rect, border_radius=10)
        
        # Draw text
        text_surface = font.render(text.title(), True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=rect.center)
        window.blit(text_surface, text_rect)
    
    pygame.display.flip()
    return buttons

def draw_game_over(window, scores, map_size):
    # Create semi-transparent overlay
    overlay = pygame.Surface((window.get_width(), window.get_height()))
    overlay.fill((0, 0, 0))
    overlay.set_alpha(128)
    window.blit(overlay, (0, 0))
    
    # Setup smaller fonts for 300x300 window
    font = pygame.font.Font(None, 48)  # Reduced from 74
    small_font = pygame.font.Font(None, 32)  # Reduced from 50
    
    # Draw "Game Over" text higher up
    game_over_text = font.render("Game Over!", True, (255, 255, 255))
    text_rect = game_over_text.get_rect(center=(window.get_width()//2, window.get_height()//4))  # Changed from //3
    window.blit(game_over_text, text_rect)
    
    # Start scores closer to "Game Over" text with smaller spacing
    y_offset = window.get_height()//3  # Changed from //2 - 50
    colors = {
        'human': (0, 0, 255),
        'policy': (255, 0, 0),
        'rl': (0, 255, 0),
    }
    
    # Find winner
    winner = max(scores.items(), key=lambda x: x[1])
    
    for player, score in scores.items():
        score_text = small_font.render(f"{player}: {score}", True, colors[player])
        text_rect = score_text.get_rect(center=(window.get_width()//2, y_offset))
        window.blit(score_text, text_rect)
        y_offset += 30  # Reduced from 50
    
    # Draw winner announcement
    winner_text = small_font.render(f"Winner: {winner[0]}!", True, colors[winner[0]])
    text_rect = winner_text.get_rect(center=(window.get_width()//2, y_offset))
    window.blit(winner_text, text_rect)
    
    # Draw smaller "New Game" button
    button_color = (0, 200, 0)
    button_rect = pygame.Rect(0, 0, 160, 40)  # Reduced from 200, 50
    button_rect.center = (window.get_width()//2, y_offset + 50)  # Reduced from 70
    pygame.draw.rect(window, button_color, button_rect, border_radius=8)
    
    button_text = small_font.render("New Game", True, (255, 255, 255))
    text_rect = button_text.get_rect(center=button_rect.center)
    window.blit(button_text, text_rect)
    
    pygame.display.flip()
    return button_rect

def main():
    # Create environment
    map_size = (300, 300)
    env = CoinCollectionEnv(map_size=map_size, render_mode='human')
    window = env.window  # Get the pygame window from environment
    rl_bot = RLBot()
    rl_bot.load('best_model_red_coins.pth')  # Load trained model
    
    running = True
    while running:
        bots = {
            # 'policy': PolicyBot(),
            'rl': rl_bot,
            'policy': HumanBot(),
            'human': None
        }
        
        # Game loop
        observation, info = env.reset()
        done = False
        clock = pygame.time.Clock()
        paused = False
        
        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    paused = True
                    buttons = draw_pause_menu(window)
                    
                    while paused and running:
                        for pause_event in pygame.event.get():
                            if pause_event.type == pygame.QUIT:
                                paused = False
                                done = True
                                running = False
                            elif pause_event.type == pygame.MOUSEBUTTONDOWN:
                                mouse_pos = pygame.mouse.get_pos()
                                if buttons['continue'].collidepoint(mouse_pos):
                                    paused = False
                                elif buttons['leave'].collidepoint(mouse_pos):
                                    paused = False
                                    done = True
                                    running = False
                            elif pause_event.type == pygame.KEYDOWN and pause_event.key == pygame.K_ESCAPE:
                                paused = False
                        
                        clock.tick(30)
                    
                    if not paused:  # Redraw the game screen when unpausing
                        env.render()
            
            if not paused:
                # Get actions for all bots
                actions = {}
                for bot_name, bot in bots.items():
                    if bot_name == 'human':
                        human_action = handle_human_input()
                        if human_action is not None:
                            actions[bot_name] = human_action
                    else:
                        actions[bot_name] = bot.get_action(observation)
                
                # Step environment with all actions
                observation, reward, done, truncated, info = env.step(actions)
                env.render()
                
                # Maintain constant frame rate
                clock.tick(50)
                
                if done and running:
                    # Show game over screen and wait for input
                    button_rect = draw_game_over(window, env.player_scores, map_size)
                    waiting_for_input = True
                    
                    while waiting_for_input and running:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                waiting_for_input = False
                                running = False
                            elif event.type == pygame.MOUSEBUTTONDOWN:
                                mouse_pos = pygame.mouse.get_pos()
                                if button_rect.collidepoint(mouse_pos):
                                    waiting_for_input = False
    
    pygame.quit()

if __name__ == "__main__":
    main() 