import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 9)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class TicTacToeAI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net_A = DQN().to(self.device)
        self.policy_net_B = DQN().to(self.device)
        
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.policy_net_A.load_state_dict(checkpoint['policy_net_A_state_dict'])
        self.policy_net_B.load_state_dict(checkpoint['policy_net_B_state_dict'])

    def _valid_actions(self, state):
        return [i for i in range(9) if state[i] == 0]

    def _get_reward(self, state, player):
        win_states = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for (i, j, k) in win_states:
            if state[i] == state[j] == state[k] and state[i] != 0:
                return 1 if state[i] == player else -1
        return 0 if 0 in state else 0.5

    def _get_ai_action(self, state, player):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net_A(state_tensor) if player == 1 else self.policy_net_B(state_tensor)
            valid_q_values = q_values.clone()
            valid_q_values[0, [i for i in range(9) if i not in self._valid_actions(state)]] = float('-inf')
            return valid_q_values.max(1)[1].item()

def create_board_buttons(state, valid_moves):
    symbols = {0: "", 1: "X", -1: "O"}
    cols = st.columns(3)
    buttons = []
    for i in range(9):
        col_idx = i % 3
        with cols[col_idx]:
            if state[i] == 0 and i in valid_moves:
                button = st.button(f"{symbols[state[i]]}", key=f"button_{i}", 
                                 use_container_width=True, 
                                 help=f"Position {i}")
            else:
                button = st.button(f"{symbols[state[i]]}", key=f"button_{i}", 
                                 use_container_width=True, 
                                 disabled=True)
            buttons.append(button)
    return buttons

def main():
    st.title("Tic-tac-toe AI Game")
    
    # Initialize game state in session state
    if 'game_state' not in st.session_state:
        st.session_state.game_state = {
            'board': np.zeros(9, dtype=int),
            'current_player': 1,
            'game_over': False,
            'message': "Game started! You are X"
        }

    # Initialize AI
    game = TicTacToeAI()
    
    # Add a file uploader for the model
    model_file = st.file_uploader("Upload AI model file (tictactoe_model.pth)", type=['pth'])
    
    if model_file is not None:
        # Save the uploaded file temporarily
        with open("temp_model.pth", "wb") as f:
            f.write(model_file.getvalue())
        game.load_model("temp_model.pth")
        os.remove("temp_model.pth")  # Clean up the temporary file
        
        # Game controls
        if st.button("Reset Game"):
            st.session_state.game_state = {
                'board': np.zeros(9, dtype=int),
                'current_player': 1,
                'game_over': False,
                'message': "Game started! You are X"
            }
        
        # Display current game status
        st.write(st.session_state.game_state['message'])
        
        # Get valid moves
        valid_moves = game._valid_actions(st.session_state.game_state['board'])
        
        # Create the game board
        buttons = create_board_buttons(st.session_state.game_state['board'], valid_moves)
        
        # Handle player moves
        if not st.session_state.game_state['game_over']:
            # Human turn
            if st.session_state.game_state['current_player'] == 1:
                for i, clicked in enumerate(buttons):
                    if clicked and i in valid_moves:
                        st.session_state.game_state['board'][i] = 1
                        reward = game._get_reward(st.session_state.game_state['board'], 1)
                        
                        if reward == 1:
                            st.session_state.game_state['game_over'] = True
                            st.session_state.game_state['message'] = "You win!"
                            st.rerun()
                        elif reward == 0.5:
                            st.session_state.game_state['game_over'] = True
                            st.session_state.game_state['message'] = "It's a draw!"
                            st.rerun()
                        else:
                            st.session_state.game_state['current_player'] = -1
                            st.session_state.game_state['message'] = "AI's turn..."
                            st.rerun()
            
            # AI turn
            else:
                action = game._get_ai_action(st.session_state.game_state['board'], -1)
                st.session_state.game_state['board'][action] = -1
                reward = game._get_reward(st.session_state.game_state['board'], -1)
                
                if reward == 1:
                    st.session_state.game_state['game_over'] = True
                    st.session_state.game_state['message'] = "AI wins!"
                    st.rerun()
                elif reward == 0.5:
                    st.session_state.game_state['game_over'] = True
                    st.session_state.game_state['message'] = "It's a draw!"
                    st.rerun()
                else:
                    st.session_state.game_state['current_player'] = 1
                    st.session_state.game_state['message'] = "Your turn!"
                    st.rerun()
    else:
        st.write("Please upload the AI model file to start playing!")

if __name__ == "__main__":
    main()
