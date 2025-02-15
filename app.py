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
    symbols = {0: "　", 1: "❌", -1: "⭕"}
    
    # カスタムCSS
    st.markdown("""
        <style>
        .board-container {
            max-width: 300px;
            margin: 0 auto;
        }
        .stButton button {
            width: 60px !important;
            height: 60px !important;
            font-size: 24px !important;
            font-weight: bold !important;
            padding: 0px !important;
            margin: 2px !important;
        }
        .board-row {
            display: flex;
            justify-content: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # ボードを3x3のグリッドで表示
    buttons = []
    for row in range(3):
        # 各行のコンテナを作成
        st.markdown('<div class="board-row">', unsafe_allow_html=True)
        cols = st.columns(3)
        for col in range(3):
            i = row * 3 + col
            with cols[col]:
                if state[i] == 0 and i in valid_moves:
                    button = st.button(f"{symbols[state[i]]}", key=f"button_{i}")
                else:
                    button = st.button(f"{symbols[state[i]]}", key=f"button_{i}", 
                                     disabled=True)
                buttons.append(button)
        st.markdown('</div>', unsafe_allow_html=True)
    
    return buttons

def initialize_game(human_first, game_ai=None):
    initial_state = {
        'board': np.zeros(9, dtype=int),
        'current_player': 1,
        'game_over': False,
        'message': "Game started! You are X" if human_first else "Game started! You are O",
        'human_symbol': 1 if human_first else -1
    }
    
    if not human_first and game_ai is not None:
        action = game_ai._get_ai_action(initial_state['board'], 1)
        initial_state['board'][action] = 1
        initial_state['current_player'] = -1
        initial_state['message'] = "Your turn!"
    
    return initial_state

def main():
    st.title("TicTacToe AI Game")
    
    # 画面の中央に配置するためのコンテナ
    st.markdown('<div class="board-container">', unsafe_allow_html=True)
    
    # Initialize AI and load model
    game = TicTacToeAI()
    model_path = "tictactoe_model.pth"
    game.load_model(model_path)
    
    # Add player order selection in sidebar
    st.sidebar.title("Game Settings")
    human_first = st.sidebar.radio(
        "Choose your role:",
        ["Play as X (First)", "Play as O (Second)"],
        index=0
    ) == "Play as X (First)"
    
    # Initialize or reset game state
    if 'game_state' not in st.session_state or st.sidebar.button("Reset Game"):
        st.session_state.game_state = initialize_game(human_first, game)
    
    # Display current game status
    st.write(st.session_state.game_state['message'])
    
    # Get valid moves
    valid_moves = game._valid_actions(st.session_state.game_state['board'])
    
    # Create the game board
    buttons = create_board_buttons(st.session_state.game_state['board'], valid_moves)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle game moves
    if not st.session_state.game_state['game_over']:
        current_player = st.session_state.game_state['current_player']
        is_human_turn = (current_player == st.session_state.game_state['human_symbol'])
        
        if is_human_turn:
            # Human turn
            for i, clicked in enumerate(buttons):
                if clicked and i in valid_moves:
                    st.session_state.game_state['board'][i] = current_player
                    reward = game._get_reward(st.session_state.game_state['board'], current_player)
                    
                    if reward == 1:
                        st.session_state.game_state['game_over'] = True
                        st.session_state.game_state['message'] = "You win!"
                        st.rerun()
                    elif reward == 0.5:
                        st.session_state.game_state['game_over'] = True
                        st.session_state.game_state['message'] = "It's a draw!"
                        st.rerun()
                    else:
                        st.session_state.game_state['current_player'] *= -1
                        st.session_state.game_state['message'] = "AI's turn..."
                        st.rerun()
        else:
            # AI turn
            action = game._get_ai_action(st.session_state.game_state['board'], current_player)
            st.session_state.game_state['board'][action] = current_player
            reward = game._get_reward(st.session_state.game_state['board'], current_player)
            
            if reward == 1:
                st.session_state.game_state['game_over'] = True
                st.session_state.game_state['message'] = "AI wins!"
                st.rerun()
            elif reward == 0.5:
                st.session_state.game_state['game_over'] = True
                st.session_state.game_state['message'] = "It's a draw!"
                st.rerun()
            else:
                st.session_state.game_state['current_player'] *= -1
                st.session_state.game_state['message'] = "Your turn!"
                st.rerun()

if __name__ == "__main__":
    main()
