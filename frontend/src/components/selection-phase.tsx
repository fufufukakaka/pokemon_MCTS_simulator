"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { PokemonData } from "@/types/battle";
import { cn } from "@/lib/utils";

interface SelectionPhaseProps {
  playerTeam: PokemonData[];
  opponentPreview: { name: string; index: number; types: string[] }[];
  onSelect: (indices: number[]) => void;
}

export function SelectionPhase({
  playerTeam,
  opponentPreview,
  onSelect,
}: SelectionPhaseProps) {
  const [selectedIndices, setSelectedIndices] = useState<number[]>([]);

  const toggleSelection = (index: number) => {
    if (selectedIndices.includes(index)) {
      setSelectedIndices(selectedIndices.filter((i) => i !== index));
    } else if (selectedIndices.length < 3) {
      setSelectedIndices([...selectedIndices, index]);
    }
  };

  const handleConfirm = () => {
    if (selectedIndices.length === 3) {
      onSelect(selectedIndices);
    }
  };

  return (
    <div className="space-y-6">
      {/* Opponent Preview - Show all 6 */}
      <Card className="border-red-200 bg-red-50/50">
        <CardHeader>
          <CardTitle className="text-red-800">相手のパーティ（6匹）</CardTitle>
          <CardDescription>
            相手のパーティを確認して、選出を決めましょう。AIも選出ネットワークであなたのパーティを見て3匹を選びます。
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            {opponentPreview.map((poke) => (
              <div
                key={poke.index}
                className="text-center p-3 bg-white rounded-lg border shadow-sm"
              >
                <div className="font-bold text-base">{poke.name}</div>
                {poke.name === "アルセウス" && (
                  <div className="flex flex-wrap gap-1 mt-1 justify-center">
                    {poke.types.map((type) => (
                      <Badge key={type} variant="outline" className="text-xs">
                        {type}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Player Team Selection */}
      <Card className="border-blue-200 bg-blue-50/50">
        <CardHeader>
          <CardTitle className="text-blue-800">
            あなたの選出（{selectedIndices.length}/3）
          </CardTitle>
          <CardDescription>
            3匹を選んでください。選んだ順番が先発順になります。
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {playerTeam.map((poke, index) => {
              const isSelected = selectedIndices.includes(index);
              const selectionOrder = selectedIndices.indexOf(index) + 1;

              return (
                <Card
                  key={index}
                  className={cn(
                    "cursor-pointer transition-all bg-white",
                    isSelected
                      ? "ring-2 ring-blue-500 bg-blue-100"
                      : "hover:bg-blue-50 hover:shadow-md"
                  )}
                  onClick={() => toggleSelection(index)}
                >
                  <CardContent className="p-3 relative">
                    {isSelected && (
                      <div className="absolute top-2 right-2 w-7 h-7 rounded-full bg-blue-500 text-white flex items-center justify-center text-sm font-bold shadow">
                        {selectionOrder}
                      </div>
                    )}
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
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Selection Summary */}
      {selectedIndices.length > 0 && (
        <Card>
          <CardContent className="py-4">
            <div className="flex items-center gap-4 flex-wrap">
              <span className="font-medium">選出:</span>
              {selectedIndices.map((idx, order) => (
                <Badge key={idx} variant="secondary" className="text-sm">
                  {order + 1}. {playerTeam[idx]?.name}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Confirm Button */}
      <Button
        size="lg"
        className="w-full"
        onClick={handleConfirm}
        disabled={selectedIndices.length !== 3}
      >
        この選出で対戦開始
      </Button>
    </div>
  );
}
