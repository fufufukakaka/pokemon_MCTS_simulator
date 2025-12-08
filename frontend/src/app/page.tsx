"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { TeamSetup } from "@/components/team-setup";
import { SelectionPhase } from "@/components/selection-phase";
import { BattleField } from "@/components/battle-field";
import {
  getTrainers,
  getPlayerParty,
  createBattle,
  selectPokemon,
  performAction,
  surrenderBattle,
} from "@/lib/api";
import type { BattleState, PokemonData, TrainerInfo, Action } from "@/types/battle";
import { Loader2 } from "lucide-react";

type GamePhase = "setup" | "selection" | "battle" | "finished";

export default function Home() {
  const [phase, setPhase] = useState<GamePhase>("setup");
  const [trainers, setTrainers] = useState<TrainerInfo[]>([]);
  const [playerParty, setPlayerParty] = useState<PokemonData[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [battleState, setBattleState] = useState<BattleState | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [aiThinkingTime, setAiThinkingTime] = useState<number | undefined>();

  // Load trainers and player party on mount
  useEffect(() => {
    Promise.all([getTrainers(), getPlayerParty()])
      .then(([trainersData, partyData]) => {
        setTrainers(trainersData.trainers);
        setPlayerParty(partyData.party);
      })
      .catch((err) => {
        console.error("Failed to load data:", err);
        setError("データの読み込みに失敗しました。バックエンドサーバーが起動しているか確認してください。");
      });
  }, []);

  const handleStartBattle = async (team: PokemonData[], aiTrainerIndex?: number) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await createBattle(team, aiTrainerIndex);
      setSessionId(response.session_id);
      setBattleState(response.state);
      setPhase("selection");
    } catch (err) {
      setError(err instanceof Error ? err.message : "バトル作成に失敗しました");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelection = async (indices: number[]) => {
    if (!sessionId) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await selectPokemon(sessionId, indices);
      setBattleState(response.state);
      setPhase("battle");
    } catch (err) {
      setError(err instanceof Error ? err.message : "選出に失敗しました");
    } finally {
      setIsLoading(false);
    }
  };

  const handleAction = async (action: Action) => {
    if (!sessionId) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await performAction(sessionId, action);
      setBattleState(response.state);
      setAiThinkingTime(response.ai_thinking_time);

      if (response.state.phase === "finished") {
        setPhase("finished");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "行動に失敗しました");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSurrender = async () => {
    if (!sessionId) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await surrenderBattle(sessionId);
      setBattleState(response.state);
      setPhase("finished");
    } catch (err) {
      setError(err instanceof Error ? err.message : "降参に失敗しました");
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setPhase("setup");
    setSessionId(null);
    setBattleState(null);
    setError(null);
    setAiThinkingTime(undefined);
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Pokemon Battle vs ReBeL AI
          </h1>
          <p className="text-gray-600 dark:text-gray-300">
            ReBeL AIとポケモンバトルで対戦しよう
          </p>
        </div>

        {/* Error display */}
        {error && (
          <Card className="mb-4 border-red-500 bg-red-50">
            <CardContent className="py-4">
              <p className="text-red-700">{error}</p>
              <Button
                variant="outline"
                size="sm"
                className="mt-2"
                onClick={() => setError(null)}
              >
                閉じる
              </Button>
            </CardContent>
          </Card>
        )}

        {/* Loading overlay */}
        {isLoading && phase === "setup" && (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-primary" />
            <span className="ml-2">読み込み中...</span>
          </div>
        )}

        {/* Setup Phase */}
        {phase === "setup" && !isLoading && (
          <div className="max-w-4xl mx-auto">
            <TeamSetup trainers={trainers} playerParty={playerParty} onStartBattle={handleStartBattle} />
          </div>
        )}

        {/* Selection Phase */}
        {phase === "selection" && battleState && (
          <div className="max-w-4xl mx-auto">
            <SelectionPhase
              playerTeam={battleState.player_team}
              opponentPreview={battleState.opponent_team_preview || []}
              onSelect={handleSelection}
            />
          </div>
        )}

        {/* Battle Phase */}
        {(phase === "battle" || phase === "finished") && battleState && (
          <div className="max-w-6xl mx-auto">
            <BattleField
              state={battleState}
              onAction={handleAction}
              onSurrender={handleSurrender}
              isLoading={isLoading}
              aiThinkingTime={aiThinkingTime}
            />

            {phase === "finished" && (
              <div className="mt-6 text-center">
                <Button size="lg" onClick={handleReset}>
                  新しいバトルを始める
                </Button>
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
