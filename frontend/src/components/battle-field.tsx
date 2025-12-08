"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { PokemonCard } from "@/components/pokemon-card";
import { ActionPanel } from "@/components/action-panel";
import { BattleLog } from "@/components/battle-log";
import type { BattleState, Action } from "@/types/battle";
import { Loader2, Clock } from "lucide-react";

interface BattleFieldProps {
  state: BattleState;
  onAction: (action: Action) => Promise<void>;
  onSurrender: () => Promise<void>;
  isLoading: boolean;
  aiThinkingTime?: number;
}

// Format seconds to MM:SS
function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

export function BattleField({
  state,
  onAction,
  onSurrender,
  isLoading,
  aiThinkingTime,
}: BattleFieldProps) {
  const [pendingAction, setPendingAction] = useState<Action | null>(null);
  const [displayTime, setDisplayTime] = useState<number>(
    state.remaining_seconds ?? 600
  );

  // Update timer every second
  useEffect(() => {
    if (state.remaining_seconds !== undefined) {
      setDisplayTime(state.remaining_seconds);
    }
  }, [state.remaining_seconds]);

  useEffect(() => {
    if (state.phase === "finished") return;

    const interval = setInterval(() => {
      setDisplayTime((prev) => Math.max(0, prev - 1));
    }, 1000);

    return () => clearInterval(interval);
  }, [state.phase]);

  const handleAction = async (action: Action) => {
    setPendingAction(action);
    try {
      await onAction(action);
    } finally {
      setPendingAction(null);
    }
  };

  const isFinished = state.phase === "finished";
  const isPlayerTurn = !isLoading && !isFinished;
  const isLowTime = displayTime < 60; // Less than 1 minute

  return (
    <div className="space-y-4">
      {/* Turn indicator and Timer */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Badge variant="outline" className="text-lg px-4 py-1">
            ターン {state.turn}
          </Badge>
          {/* TOD Timer */}
          {!isFinished && (
            <div
              className={`flex items-center gap-2 px-3 py-1 rounded-md ${
                isLowTime
                  ? "bg-red-100 text-red-700 animate-pulse"
                  : "bg-gray-100 text-gray-700"
              }`}
            >
              <Clock className="w-4 h-4" />
              <span className="font-mono font-bold">
                {formatTime(displayTime)}
              </span>
            </div>
          )}
        </div>
        <div className="flex items-center gap-4">
          {isLoading && (
            <div className="flex items-center gap-2 text-muted-foreground">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>AIが考え中...</span>
            </div>
          )}
          {aiThinkingTime !== undefined && aiThinkingTime > 0 && !isLoading && (
            <Badge variant="secondary">
              AI思考時間: {aiThinkingTime.toFixed(2)}s
            </Badge>
          )}
          {!isFinished && (
            <Button
              variant="destructive"
              size="sm"
              onClick={onSurrender}
              disabled={isLoading}
            >
              降参
            </Button>
          )}
        </div>
      </div>

      {/* Battle result */}
      {isFinished && (
        <Card className={state.winner === 0 ? "bg-green-50" : "bg-red-50"}>
          <CardContent className="py-6 text-center">
            <h2 className="text-2xl font-bold">
              {state.winner === 0 ? "勝利！" : "敗北..."}
            </h2>
          </CardContent>
        </Card>
      )}

      {/* Field */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left: Player side */}
        <div className="space-y-4">
          <Card>
            <CardHeader className="py-2">
              <CardTitle className="text-sm text-muted-foreground">
                あなたのポケモン
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {/* Active Pokemon */}
              <PokemonCard
                pokemon={state.player_active}
                isPlayer={true}
                isActive={true}
                showDetails={true}
              />

              {/* Bench */}
              <div className="grid grid-cols-2 gap-2">
                {state.player_bench.map((pokemon, i) => (
                  <PokemonCard
                    key={i}
                    pokemon={pokemon}
                    isPlayer={true}
                    showDetails={true}
                  />
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Field conditions - player side */}
          {state.field && (
            <FieldConditions
              side={state.field.player_side}
              label="味方フィールド"
            />
          )}
        </div>

        {/* Center: Actions & Log */}
        <div className="space-y-4">
          {/* Weather/Terrain */}
          {state.field && (state.field.weather || state.field.terrain) && (
            <Card>
              <CardContent className="py-2 flex gap-2 justify-center">
                {state.field.weather && (
                  <Badge>{state.field.weather}</Badge>
                )}
                {state.field.terrain && (
                  <Badge variant="outline">{state.field.terrain}</Badge>
                )}
              </CardContent>
            </Card>
          )}

          {/* Change phase - select next pokemon */}
          {state.phase === "change" && isPlayerTurn && (
            <Card className="bg-yellow-50 border-yellow-300">
              <CardHeader className="py-3">
                <CardTitle className="text-lg text-yellow-800">
                  次のポケモンを選んでください
                </CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-2">
                {state.available_actions
                  .filter((a) => a.type === "switch")
                  .map((action) => {
                    const benchPokemon = state.player_bench.find(
                      (p) => p?.name === action.name
                    );
                    return (
                      <Button
                        key={`change-${action.index}`}
                        onClick={() => handleAction(action)}
                        disabled={isLoading || !!pendingAction}
                        className="h-auto py-3 bg-yellow-500 hover:bg-yellow-600 text-white"
                      >
                        <div className="text-center w-full">
                          <div className="font-bold">{action.name}</div>
                          {benchPokemon && (
                            <div className="text-xs opacity-90">
                              HP: {Math.round(benchPokemon.hp_ratio * 100)}%
                            </div>
                          )}
                        </div>
                      </Button>
                    );
                  })}
              </CardContent>
            </Card>
          )}

          {/* Actions - normal battle phase */}
          {isPlayerTurn && state.phase === "battle" && (
            <ActionPanel
              actions={state.available_actions}
              onAction={handleAction}
              disabled={isLoading || !!pendingAction}
              activePokemon={state.player_active}
              bench={state.player_bench}
            />
          )}

          {/* Battle Log */}
          <BattleLog log={state.log} />
        </div>

        {/* Right: Opponent side */}
        <div className="space-y-4">
          <Card>
            <CardHeader className="py-2">
              <CardTitle className="text-sm text-muted-foreground">
                相手のポケモン
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {/* Active Pokemon */}
              <PokemonCard
                pokemon={state.opponent_active}
                isPlayer={false}
                isActive={true}
              />

              {/* Bench (opponent - limited info) */}
              <div className="grid grid-cols-2 gap-2">
                {state.opponent_bench.map((pokemon, i) => (
                  <PokemonCard key={i} pokemon={pokemon} isPlayer={false} />
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Field conditions - opponent side */}
          {state.field && (
            <FieldConditions
              side={state.field.opponent_side}
              label="相手フィールド"
            />
          )}
        </div>
      </div>
    </div>
  );
}

interface FieldConditionsProps {
  side: {
    stealth_rock: boolean;
    spikes: number;
    toxic_spikes: number;
    sticky_web: boolean;
    reflect: number;
    light_screen: number;
    tailwind: number;
  };
  label: string;
}

function FieldConditions({ side, label }: FieldConditionsProps) {
  const conditions = [];

  if (side.stealth_rock) conditions.push("ステルスロック");
  if (side.spikes > 0) conditions.push(`まきびし×${side.spikes}`);
  if (side.toxic_spikes > 0) conditions.push(`どくびし×${side.toxic_spikes}`);
  if (side.sticky_web) conditions.push("ねばねばネット");
  if (side.reflect > 0) conditions.push(`リフレクター(${side.reflect})`);
  if (side.light_screen > 0) conditions.push(`ひかりのかべ(${side.light_screen})`);
  if (side.tailwind > 0) conditions.push(`おいかぜ(${side.tailwind})`);

  if (conditions.length === 0) return null;

  return (
    <Card>
      <CardContent className="py-2">
        <div className="text-xs text-muted-foreground mb-1">{label}</div>
        <div className="flex flex-wrap gap-1">
          {conditions.map((cond) => (
            <Badge key={cond} variant="secondary" className="text-xs">
              {cond}
            </Badge>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
