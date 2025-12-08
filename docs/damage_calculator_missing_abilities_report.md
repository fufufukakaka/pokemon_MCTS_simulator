# Damage Calculator API: Missing Abilities Analysis Report

## Executive Summary

After analyzing the battle simulation engine (`battle.py`) and the damage calculator API (`stat_calculator.py`), I've identified **significant gaps** in ability implementations between the two systems. The battle engine implements approximately **150+ abilities** affecting damage calculations, while the damage calculator API currently implements only **~20 abilities**. This report categorizes the missing abilities by their impact type and provides implementation priority recommendations.

## Current Implementation Status

### Damage Calculator API (Currently Implemented)
The damage calculator API (`stat_calculator.py`) currently implements:

**Attack Modifying Abilities:**
- ちからもち (Pure Power) - 2x physical attack
- ヨガパワー (Huge Power) - 2x physical attack  
- こんじょう (Guts) - 1.5x physical attack when statused
- サンパワー (Solar Power) - 1.5x special attack in sun
- ソーラーパワー (Solar Power) - 1.5x special attack in sun
- はりきり (Hustle) - 1.5x physical attack
- もうか (Blaze) - 1.5x fire moves when HP ≤ 1/3
- しんりょく (Overgrow) - 1.5x grass moves when HP ≤ 1/3
- げきりゅう (Torrent) - 1.5x water moves when HP ≤ 1/3
- むしのしらせ (Swarm) - 1.5x bug moves when HP ≤ 1/3
- ひひいろのこどう (Orichalcum Pulse) - 1.33x attack in sun

**Defense Modifying Abilities:**
- ファーコート (Fur Coat) - 2x physical defense
- マルチスケイル (Multiscale) - 0.5x damage at full HP
- シャドーシールド (Shadow Shield) - 0.5x damage at full HP
- ふしぎなうろこ (Marvel Scale) - 1.5x defense when statused

**Disaster Abilities:**
- わざわいのうつわ (Vessel of Ruin) - 0.75x opponent special attack
- わざわいのおふだ (Tablet of Ruin) - 0.75x opponent physical attack
- わざわいのつるぎ (Sword of Ruin) - 0.75x opponent physical defense
- わざわいのたま (Beads of Ruin) - 0.75x opponent special defense

## Missing Abilities Analysis

### HIGH PRIORITY - Core Damage Calculation Abilities

#### Attack Stat Modifiers
| Ability | Effect | Implementation Priority |
|---------|---------|------------------------|
| **いわはこび** (Solid Rock) | 1.5x rock move power | **HIGH** |
| **すいほう** (Water Bubble) | 2x water move power | **HIGH** |
| **トランジスタ** (Transistor) | 1.3x electric move power | **HIGH** |
| **はがねつかい** (Steelworker) | 1.5x steel move power | **HIGH** |
| **はがねのせいしん** (Steel Spirit) | 1.5x steel move power | **HIGH** |
| **りゅうのあぎと** (Dragon's Maw) | 1.5x dragon move power | **HIGH** |
| **ハドロンエンジン** (Hadron Engine) | 1.33x special attack in Electric Terrain | **HIGH** |
| **フラワーギフト** (Flower Gift) | 1.5x attack in sun | **HIGH** |
| **ごりむちゅう** (Gorilla Tactics) | 1.5x physical attack (choice locked) | **HIGH** |
| **ねつぼうそう** (Flare Boost) | 1.5x special attack when burned | **HIGH** |
| **どくぼうそう** (Poison Touch) | 1.5x physical attack when poisoned | **HIGH** |

#### Defense Stat Modifiers
| Ability | Effect | Implementation Priority |
|---------|---------|------------------------|
| **くさのけがわ** (Grass Pelt) | 1.5x defense in Grassy Terrain | **HIGH** |
| **あついしぼう** (Thick Fat) | 0.5x fire/ice damage | **HIGH** |
| **きよめのしお** (Purifying Salt) | 0.5x ghost damage | **HIGH** |

#### Damage Multiplier Abilities
| Ability | Effect | Implementation Priority |
|---------|---------|------------------------|
| **いろめがね** (Tinted Lens) | 2x damage on not very effective moves | **HIGH** |
| **スナイパー** (Sniper) | 1.5x critical hit damage | **HIGH** |
| **ハードロック** (Solid Rock) | 0.75x super effective damage | **HIGH** |
| **フィルター** (Filter) | 0.75x super effective damage | **HIGH** |
| **プリズムアーマー** (Prism Armor) | 0.75x super effective damage | **HIGH** |

### MEDIUM PRIORITY - Situational Damage Modifiers

#### Power Modifying Abilities
| Ability | Effect | Implementation Priority |
|---------|---------|------------------------|
| **テクニシャン** (Technician) | 1.5x moves with ≤60 power | **MEDIUM** |
| **アナライズ** (Analytic) | 1.3x when moving last | **MEDIUM** |
| **かたいつめ** (Tough Claws) | 1.3x contact moves | **MEDIUM** |
| **がんじょうあご** (Strong Jaw) | 1.5x biting moves | **MEDIUM** |
| **きれあじ** (Sharpness) | 1.5x cutting moves | **MEDIUM** |
| **すなのちから** (Sand Force) | 1.3x rock/ground/steel in sandstorm | **MEDIUM** |
| **そうだいしょう** (Supreme Overlord) | Power boost based on fainted team members | **MEDIUM** |
| **ちからずく** (Sheer Force) | 1.3x moves with secondary effects | **MEDIUM** |
| **てつのこぶし** (Iron Fist) | 1.2x punching moves | **MEDIUM** |
| **とうそうしん** (Rivalry) | 1.25x vs same gender, 0.75x vs opposite | **MEDIUM** |
| **パンクロック** (Punk Rock) | 1.3x sound moves | **MEDIUM** |
| **メガランチャー** (Mega Launcher) | 1.5x pulse moves | **MEDIUM** |

#### Skin Abilities (Type Change + Power Boost)
| Ability | Effect | Implementation Priority |
|---------|---------|------------------------|
| **エレキスキン** (Electric Surge) | Normal → Electric + 1.2x power | **MEDIUM** |
| **スカイスキン** (Aerilate) | Normal → Flying + 1.2x power | **MEDIUM** |
| **フェアリースキン** (Pixilate) | Normal → Fairy + 1.2x power | **MEDIUM** |
| **フリーズスキン** (Refrigerate) | Normal → Ice + 1.2x power | **MEDIUM** |
| **ノーマルスキン** (Normalize) | All moves → Normal + 1.2x power | **MEDIUM** |

#### Weather/Terrain Power Modifiers
| Ability | Effect | Implementation Priority |
|---------|---------|------------------------|
| **かんそうはだ** (Dry Skin) | 1.25x fire damage taken, water immunity | **MEDIUM** |
| **たいねつ** (Heatproof) | 0.5x fire damage | **MEDIUM** |

### MEDIUM PRIORITY - Defensive Abilities

#### Damage Immunity/Reduction
| Ability | Effect | Implementation Priority |
|---------|---------|------------------------|
| **かぜのり** (Wind Rider) | Wind move immunity | **MEDIUM** |
| **こおりのりんぷん** (Ice Scales) | 0.5x special damage | **MEDIUM** |
| **こんがりボディ** (Well-Baked Body) | Fire immunity | **MEDIUM** |
| **そうしょく** (Sap Sipper) | Grass immunity | **MEDIUM** |
| **ちくでん** (Volt Absorb) | Electric immunity | **MEDIUM** |
| **でんきエンジン** (Motor Drive) | Electric immunity | **MEDIUM** |
| **ひらいしん** (Lightning Rod) | Electric immunity | **MEDIUM** |
| **ちょすい** (Water Absorb) | Water immunity | **MEDIUM** |
| **よびみず** (Storm Drain) | Water immunity | **MEDIUM** |
| **どしょく** (Earth Eater) | Ground immunity | **MEDIUM** |
| **パンクロック** (Punk Rock) | 0.5x sound move damage | **MEDIUM** |
| **ぼうおん** (Soundproof) | Sound move immunity | **MEDIUM** |
| **ぼうだん** (Bulletproof) | Bullet/ball move immunity | **MEDIUM** |
| **ファントムガード** (Phantom Guard) | 0.5x damage at full HP | **MEDIUM** |
| **もふもふ** (Fluffy) | 0.5x contact, 2x fire damage | **MEDIUM** |

### LOW PRIORITY - Special Case Abilities

#### Stat Reduction Abilities
| Ability | Effect | Implementation Priority |
|---------|---------|------------------------|
| **スロースタート** (Slow Start) | 0.5x attack/speed for 5 turns | **LOW** |
| **よわき** (Defeatist) | 0.5x attack when HP ≤ 50% | **LOW** |

#### Aura Abilities
| Ability | Effect | Implementation Priority |
|---------|---------|------------------------|
| **ダークオーラ** (Dark Aura) | 1.33x dark moves (all Pokémon) | **LOW** |
| **フェアリーオーラ** (Fairy Aura) | 1.33x fairy moves (all Pokémon) | **LOW** |
| **オーラブレイク** (Aura Break) | Reverses aura effects | **LOW** |

#### Fire Absorption
| Ability | Effect | Implementation Priority |
|---------|---------|------------------------|
| **もらいび** (Flash Fire) | Fire immunity + 1.5x fire power | **LOW** |

## Implementation Recommendations

### Phase 1: Core Damage Modifiers (HIGH Priority)
Focus on abilities that directly modify attack/defense stats and basic damage calculations:
1. **すいほう** (Water Bubble) - Major water-type damage modifier
2. **いわはこび** (Solid Rock), **トランジスタ** (Transistor) - Type-specific power boosts
3. **いろめがね** (Tinted Lens) - Fundamental type effectiveness modifier
4. **ハードロック/フィルター** (Solid Rock/Filter) - Super effective damage reduction
5. **あついしぼう** (Thick Fat) - Common defensive ability

### Phase 2: Situational Modifiers (MEDIUM Priority)
Implement abilities with conditional effects:
1. **テクニシャン** (Technician) - Affects many weak moves
2. **Skin abilities** - Type changing abilities with power boosts
3. **Immunity abilities** - Complete damage negation
4. **Contact/category-based** modifiers

### Phase 3: Complex Interactions (LOW Priority)
Handle abilities with turn-based effects or complex interactions:
1. **Aura abilities** - Affect multiple Pokémon
2. **Turn-based** abilities like スロースタート
3. **Absorption** abilities with state changes

## Technical Implementation Notes

### Required Infrastructure Additions
1. **Turn counter** - For abilities like スロースタート
2. **Team member tracking** - For そうだいしょう
3. **Move category checking** - For contact, sound, cutting moves
4. **Type effectiveness integration** - For いろめがね, ハードロック
5. **Gender tracking** - For とうそうしん
6. **Action order tracking** - For アナライズ

### Data Structure Enhancements
```python
# Additional fields needed in PokemonState
class PokemonState:
    gender: Optional[str]  # For rivalry
    turn_count: int = 0   # For slow start
    fainted_teammates: int = 0  # For supreme overlord
```

### Integration Points
- **Move categorization** - Need access to move category data (contact, sound, etc.)
- **Type effectiveness** - Integration with TypeCalculator
- **Weather/terrain effects** - Already partially implemented
- **Item interactions** - Some abilities interact with items

## Conclusion

The damage calculator API is missing approximately **130+ abilities** that are implemented in the battle engine. Implementing the HIGH priority abilities (15-20 abilities) would cover the most common and impactful damage calculation scenarios. The MEDIUM priority abilities (40+ abilities) would handle most situational cases, while LOW priority abilities cover edge cases and complex interactions.

**Estimated implementation effort:**
- **Phase 1 (HIGH)**: 2-3 weeks
- **Phase 2 (MEDIUM)**: 4-6 weeks  
- **Phase 3 (LOW)**: 2-3 weeks

This phased approach ensures the most critical damage calculation scenarios are handled first while building toward comprehensive ability coverage.