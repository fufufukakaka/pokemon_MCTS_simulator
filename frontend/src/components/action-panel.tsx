"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Action, BattlePokemonState } from "@/types/battle";
import { cn } from "@/lib/utils";

interface ActionPanelProps {
  actions: Action[];
  onAction: (action: Action) => void;
  disabled?: boolean;
  activePokemon?: BattlePokemonState | null;
  bench?: BattlePokemonState[];
}

const moveTypeColors: Record<string, string> = {
  ノーマル: "bg-gray-400 hover:bg-gray-500",
  ほのお: "bg-orange-500 hover:bg-orange-600",
  みず: "bg-blue-500 hover:bg-blue-600",
  でんき: "bg-yellow-400 hover:bg-yellow-500",
  くさ: "bg-green-500 hover:bg-green-600",
  こおり: "bg-cyan-400 hover:bg-cyan-500",
  かくとう: "bg-red-700 hover:bg-red-800",
  どく: "bg-purple-500 hover:bg-purple-600",
  じめん: "bg-amber-600 hover:bg-amber-700",
  ひこう: "bg-indigo-300 hover:bg-indigo-400",
  エスパー: "bg-pink-500 hover:bg-pink-600",
  むし: "bg-lime-500 hover:bg-lime-600",
  いわ: "bg-amber-700 hover:bg-amber-800",
  ゴースト: "bg-purple-700 hover:bg-purple-800",
  ドラゴン: "bg-indigo-600 hover:bg-indigo-700",
  あく: "bg-gray-700 hover:bg-gray-800",
  はがね: "bg-gray-500 hover:bg-gray-600",
  フェアリー: "bg-pink-300 hover:bg-pink-400",
};

export function ActionPanel({
  actions,
  onAction,
  disabled = false,
  activePokemon,
  bench,
}: ActionPanelProps) {
  const moveActions = actions.filter((a) => a.type === "move");
  const switchActions = actions.filter((a) => a.type === "switch");
  const teraActions = actions.filter((a) => a.type === "terastallize");

  return (
    <div className="space-y-4">
      {/* Moves */}
      <Card>
        <CardHeader className="py-3">
          <CardTitle className="text-lg">技</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-2">
          {moveActions.length > 0 ? (
            moveActions.map((action) => (
              <Button
                key={`move-${action.index}`}
                onClick={() => onAction(action)}
                disabled={disabled || action.disabled}
                className={cn(
                  "h-auto py-3 text-white",
                  "bg-primary hover:bg-primary/90"
                )}
                variant="default"
              >
                <div className="text-center">
                  <div className="font-bold">{action.name}</div>
                  {action.pp !== undefined && action.max_pp !== undefined && (
                    <div className="text-xs opacity-75">
                      PP: {action.pp}/{action.max_pp}
                    </div>
                  )}
                  {action.disabled_reason && (
                    <div className="text-xs opacity-75">
                      {action.disabled_reason}
                    </div>
                  )}
                </div>
              </Button>
            ))
          ) : (
            <div className="col-span-2 text-center text-muted-foreground py-4">
              使用可能な技がありません
            </div>
          )}
        </CardContent>
      </Card>

      {/* Terastallize */}
      {teraActions.length > 0 && (
        <Card className="border-purple-300 bg-gradient-to-r from-purple-50 to-pink-50">
          <CardHeader className="py-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-pink-600">
                テラスタル
              </span>
              {activePokemon && (
                <span className="text-sm font-normal text-muted-foreground">
                  → {(activePokemon as any).tera_type || "???"}タイプ
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-2">
            {teraActions.map((action) => (
              <Button
                key={`tera-${action.index}`}
                onClick={() => onAction(action)}
                disabled={disabled || action.disabled}
                className="h-auto py-3 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white"
              >
                <div className="text-center">
                  <div className="font-bold">{action.name}</div>
                  {action.pp !== undefined && action.max_pp !== undefined && (
                    <div className="text-xs opacity-75">
                      PP: {action.pp}/{action.max_pp}
                    </div>
                  )}
                </div>
              </Button>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Switch */}
      <Card>
        <CardHeader className="py-3">
          <CardTitle className="text-lg">交代</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-2">
          {switchActions.length > 0 ? (
            switchActions.map((action) => {
              const benchPokemon = bench?.find((p) => p?.name === action.name);
              return (
                <Button
                  key={`switch-${action.index}`}
                  onClick={() => onAction(action)}
                  disabled={disabled || action.disabled}
                  variant="outline"
                  className="h-auto py-2"
                >
                  <div className="text-center w-full">
                    <div className="font-bold">{action.name}</div>
                    {benchPokemon && (
                      <div className="text-xs text-muted-foreground">
                        HP: {Math.round(benchPokemon.hp_ratio * 100)}%
                      </div>
                    )}
                  </div>
                </Button>
              );
            })
          ) : (
            <div className="col-span-2 text-center text-muted-foreground py-4">
              交代可能なポケモンがいません
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
