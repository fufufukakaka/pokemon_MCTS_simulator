"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { PokemonData, TrainerInfo } from "@/types/battle";

interface TeamSetupProps {
  trainers: TrainerInfo[];
  playerParty: PokemonData[];
  onStartBattle: (team: PokemonData[], aiTrainerIndex?: number) => void;
}

export function TeamSetup({ trainers, playerParty, onStartBattle }: TeamSetupProps) {
  const [selectedTrainerIndex, setSelectedTrainerIndex] = useState<number | undefined>(
    undefined
  );
  const [selectedTeam, setSelectedTeam] = useState<PokemonData[]>(playerParty);

  // Update selected team when playerParty loads
  useEffect(() => {
    if (playerParty.length > 0) {
      setSelectedTeam(playerParty);
    }
  }, [playerParty]);

  const handleStart = () => {
    if (selectedTeam.length >= 3) {
      onStartBattle(selectedTeam, selectedTrainerIndex);
    }
  };

  return (
    <div className="space-y-6">
      {/* Player Party */}
      <Card>
        <CardHeader>
          <CardTitle>あなたのパーティ</CardTitle>
        </CardHeader>
        <CardContent>
          {selectedTeam.length > 0 ? (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {selectedTeam.map((poke, j) => (
                <Card key={j} className="p-3">
                  <div className="font-bold text-base">{poke.name}</div>
                  <div className="text-sm text-muted-foreground">
                    {poke.item}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {poke.ability}
                  </div>
                  <div className="flex flex-wrap gap-1 mt-2">
                    {poke.moves.map((move) => (
                      <Badge key={move} variant="outline" className="text-xs">
                        {move}
                      </Badge>
                    ))}
                  </div>
                  <Badge className="mt-2 text-xs">
                    テラス: {poke.tera_type}
                  </Badge>
                </Card>
              ))}
            </div>
          ) : (
            <div className="text-center text-muted-foreground py-8">
              パーティデータを読み込み中...
            </div>
          )}
        </CardContent>
      </Card>

      {/* AI Trainer Selection */}
      <Card>
        <CardHeader>
          <CardTitle>対戦相手</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 max-h-[300px] overflow-y-auto">
            <Button
              variant={selectedTrainerIndex === undefined ? "default" : "outline"}
              className="w-full justify-start"
              onClick={() => setSelectedTrainerIndex(undefined)}
            >
              ランダム
            </Button>
            {trainers.slice(0, 30).map((trainer) => (
              <Button
                key={trainer.index}
                variant={
                  selectedTrainerIndex === trainer.index ? "default" : "outline"
                }
                className="w-full justify-start"
                onClick={() => setSelectedTrainerIndex(trainer.index)}
              >
                <div className="text-left">
                  <div>{trainer.name || `Trainer ${trainer.index}`}</div>
                  <div className="text-xs text-muted-foreground">
                    {trainer.pokemon_names.slice(0, 6).join(", ")}
                  </div>
                </div>
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Start Button */}
      <Button
        size="lg"
        className="w-full"
        onClick={handleStart}
        disabled={selectedTeam.length < 3}
      >
        バトル開始
      </Button>
    </div>
  );
}
