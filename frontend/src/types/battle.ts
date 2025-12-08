export interface PokemonData {
  name: string;
  item: string;
  ability: string;
  moves: string[];
  tera_type: string;
  nature?: string;
  evs?: {
    hp: number;
    attack: number;
    defense: number;
    sp_attack: number;
    sp_defense: number;
    speed: number;
  };
}

export interface BattlePokemonState {
  name: string;
  hp_ratio: number; // 0-1
  max_hp: number;
  current_hp: number;
  status: string | null; // "まひ", "やけど", etc.
  item: string | null; // null if revealed and consumed
  is_terastallized: boolean;
  tera_type: string;
  types: string[];
  moves: string[]; // Revealed moves (for opponent)
  ability: string | null; // Revealed ability
  stat_changes: {
    attack: number;
    defense: number;
    sp_attack: number;
    sp_defense: number;
    speed: number;
  };
}

export interface FieldState {
  weather: string | null;
  terrain: string | null;
  player_side: {
    stealth_rock: boolean;
    spikes: number;
    toxic_spikes: number;
    sticky_web: boolean;
    reflect: number; // remaining turns
    light_screen: number;
    tailwind: number;
  };
  opponent_side: {
    stealth_rock: boolean;
    spikes: number;
    toxic_spikes: number;
    sticky_web: boolean;
    reflect: number;
    light_screen: number;
    tailwind: number;
  };
}

export interface BattleState {
  session_id: string;
  turn: number;
  phase: "team_preview" | "selection" | "battle" | "change" | "finished";
  winner: number | null;

  // Player's view (player 0)
  player_active: BattlePokemonState | null;
  player_bench: BattlePokemonState[];
  player_team: PokemonData[]; // Full info for player's team

  // Opponent's view (player 1 - AI)
  opponent_active: BattlePokemonState | null;
  opponent_bench: BattlePokemonState[];
  opponent_team_preview?: { name: string; index: number; types: string[] }[];

  field: FieldState;

  // Available actions
  available_actions: Action[];

  // Battle log
  log: LogEntry[];
}

export interface Action {
  type: "move" | "switch" | "terastallize";
  index: number; // move index (0-3) or pokemon index for switch
  name: string; // move name or pokemon name
  disabled?: boolean;
  disabled_reason?: string;
  tera_type?: string; // for terastallize action
  pp?: number; // current PP for move actions
  max_pp?: number; // max PP for move actions
}

export interface LogEntry {
  turn: number;
  messages: string[];
}

export interface CreateBattleRequest {
  player_team: PokemonData[];
  ai_trainer_index?: number; // Use specific trainer from data
}

export interface CreateBattleResponse {
  session_id: string;
  state: BattleState;
}

export interface SelectionRequest {
  session_id: string;
  selected_indices: number[]; // indices of selected pokemon (3)
}

export interface ActionRequest {
  session_id: string;
  action: Action;
}

export interface ActionResponse {
  state: BattleState;
  ai_action?: {
    type: string;
    name: string;
  };
  ai_thinking_time?: number;
}

export interface TrainerInfo {
  index: number;
  name: string;
  pokemon_names: string[];
}
