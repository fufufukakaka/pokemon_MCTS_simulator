"use client";

import { cn } from "@/lib/utils";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import type { BattlePokemonState } from "@/types/battle";

interface PokemonCardProps {
  pokemon: BattlePokemonState | null;
  isPlayer?: boolean;
  isActive?: boolean;
  onClick?: () => void;
  selectable?: boolean;
  selected?: boolean;
  showDetails?: boolean;
}

const typeColors: Record<string, string> = {
  ノーマル: "bg-gray-400",
  ほのお: "bg-orange-500",
  みず: "bg-blue-500",
  でんき: "bg-yellow-400",
  くさ: "bg-green-500",
  こおり: "bg-cyan-300",
  かくとう: "bg-red-700",
  どく: "bg-purple-500",
  じめん: "bg-amber-600",
  ひこう: "bg-indigo-300",
  エスパー: "bg-pink-500",
  むし: "bg-lime-500",
  いわ: "bg-amber-700",
  ゴースト: "bg-purple-700",
  ドラゴン: "bg-indigo-600",
  あく: "bg-gray-700",
  はがね: "bg-gray-500",
  フェアリー: "bg-pink-300",
};

function getHpColor(ratio: number): string {
  if (ratio > 0.5) return "bg-green-500";
  if (ratio > 0.25) return "bg-yellow-500";
  return "bg-red-500";
}

export function PokemonCard({
  pokemon,
  isPlayer = false,
  isActive = false,
  onClick,
  selectable = false,
  selected = false,
  showDetails = false,
}: PokemonCardProps) {
  if (!pokemon) {
    return (
      <Card className="w-full h-32 bg-muted/50 flex items-center justify-center">
        <span className="text-muted-foreground">Empty</span>
      </Card>
    );
  }

  const hpPercent = pokemon.hp_ratio * 100;

  return (
    <Card
      className={cn(
        "w-full transition-all",
        isActive && "ring-2 ring-primary",
        selectable && "cursor-pointer hover:bg-accent",
        selected && "ring-2 ring-green-500 bg-green-50",
        pokemon.current_hp <= 0 && "opacity-50"
      )}
      onClick={selectable ? onClick : undefined}
    >
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="font-bold text-lg">{pokemon.name}</span>
            {pokemon.is_terastallized && (
              <Badge variant="outline" className="text-xs">
                テラス: {pokemon.tera_type}
              </Badge>
            )}
          </div>
          {pokemon.status && (
            <Badge variant="destructive" className="text-xs">
              {pokemon.status}
            </Badge>
          )}
        </div>

        {/* Types */}
        <div className="flex gap-1 mb-2">
          {pokemon.types?.map((type) => (
            <Badge
              key={type}
              className={cn("text-xs text-white", typeColors[type] || "bg-gray-500")}
            >
              {type}
            </Badge>
          ))}
        </div>

        {/* HP Bar */}
        <div className="space-y-1">
          <div className="flex justify-between text-sm">
            <span>HP</span>
            <span>
              {pokemon.current_hp} / {pokemon.max_hp}
            </span>
          </div>
          <Progress
            value={hpPercent}
            className="h-3"
            indicatorClassName={getHpColor(pokemon.hp_ratio)}
          />
        </div>

        {/* Stat changes */}
        {pokemon.stat_changes && (
          <div className="flex gap-1 mt-2 flex-wrap">
            {Object.entries(pokemon.stat_changes).map(([stat, value]) => {
              if (value === 0) return null;
              const label =
                stat === "attack"
                  ? "攻"
                  : stat === "defense"
                  ? "防"
                  : stat === "sp_attack"
                  ? "特攻"
                  : stat === "sp_defense"
                  ? "特防"
                  : "速";
              return (
                <Badge
                  key={stat}
                  variant={value > 0 ? "default" : "destructive"}
                  className="text-xs"
                >
                  {label} {value > 0 ? `+${value}` : value}
                </Badge>
              );
            })}
          </div>
        )}

        {/* Item and Ability (if revealed) */}
        {showDetails && (
          <div className="mt-2 text-sm text-muted-foreground">
            {pokemon.item && <div>持ち物: {pokemon.item}</div>}
            {pokemon.ability && <div>特性: {pokemon.ability}</div>}
          </div>
        )}

        {/* Moves (if revealed) */}
        {showDetails && pokemon.moves && pokemon.moves.length > 0 && (
          <div className="mt-2">
            <div className="text-xs text-muted-foreground mb-1">技:</div>
            <div className="flex flex-wrap gap-1">
              {pokemon.moves.map((move) => (
                <Badge key={move} variant="outline" className="text-xs">
                  {move}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
