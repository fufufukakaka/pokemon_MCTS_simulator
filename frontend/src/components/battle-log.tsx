"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { LogEntry } from "@/types/battle";

interface BattleLogProps {
  log: LogEntry[];
}

export function BattleLog({ log }: BattleLogProps) {
  // Reverse log to show newest first
  const reversedLog = [...log].reverse();

  return (
    <Card className="h-full">
      <CardHeader className="py-3">
        <CardTitle className="text-lg">バトルログ</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea className="h-[300px] px-4">
          <div className="space-y-3 pb-4">
            {reversedLog.map((entry, i) => (
              <div key={i} className="space-y-1">
                <div className="text-xs text-muted-foreground font-medium">
                  ターン {entry.turn}
                </div>
                {entry.messages.map((msg, j) => (
                  <div key={j} className="text-sm pl-2 border-l-2 border-muted">
                    {msg}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
