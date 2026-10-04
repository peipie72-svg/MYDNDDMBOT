"""
Telegram-бот «Dungeon Master» для настольной ролевой игры Dungeons & Dragons 5e (2024 / «5.5e»).

Стек:
    * Python 3
    * aiogram 3.x      — асинхронный фреймворк для Telegram Bot API
    * openai (SDK)     — обращение к DeepSeek API по OpenAI-совместимому интерфейсу
    * python-dotenv    — загрузка переменных окружения из файла .env
    * sqlite3 (stdlib) — постоянное хранение сессий в файле bot_database.db

Команды:
    /start     — приветствие, сброс прошлой сессии и создание героя
    /reset     — принудительный сброс контекста, игра начинается с чистого листа
    /hero      — сгенерировать случайного героя 1-го уровня по правилам PHB 2024
    /roll      — бросок кубиков, считаемый кодом (например: /roll d20, /roll 2d6+3)
    /sheet     — показать лист персонажа (имя, класс, HP, характеристики, снаряжение)
    /inventory — список снаряжения и золота
    /check     — меню проверок характеристик (СИЛ/ЛОВ/ТЕЛ/ИНТ/МУД/ХАР)

ЭТАП 1 — создание персонажа:
    Новая игра начинается не с пролога, а с создания героя. В первом ответе Мастер
    приветствует игрока и просит имя, вид (расу), класс и краткое описание героя
    (либо «Случайный герой»). Пока герой не подтверждён, Мастер не описывает мир и
    не начинает сюжет. Характеристики, максимум HP и стартовое снаряжение 1-го уровня
    выставляет КОД бота (см. build_random_hero и apply_starter_loadout),
    а не нейросеть; Мастер лишь вносит имя, вид, класс и описание в служебном блоке.
    После подтверждения героя (сообщение «да» или кнопка «✅ Подтвердить героя»)
    начинается вводная сцена пролога.

Инлайн-кнопки (под каждым ответом Мастера):
    🎲 d20 / 🎲 d20 с преим. / 🎲 d20 с помех. — мгновенный бросок кодом
    📜 Лист / 🎒 Инвентарь / 🎲 Бросок урона    — лист, снаряжение, урон оружием
    🧠 Проверки по статам                      — меню проверок d20 + модификатор
    🎲 Случайный герой / ✅ Подтвердить героя    — кнопки этапа создания персонажа
    Любое нажатие пишется в историю и SQLite так же, как обычная команда игрока.

Любое другое текстовое сообщение воспринимается как действие игрока
и передаётся Мастеру вместе с историей диалога.

Хранение данных:
    В локальной базе bot_database.db (в корне проекта) две таблицы:
        * characters   — сериализованный лист персонажа, по одному на игрока;
        * chat_history — лог переписки (роли user / assistant).
    При первом обращении игрока загружаются его Character и последние
    MAX_HISTORY_MESSAGES сообщений; каждое изменение листа и каждое сообщение
    сразу записываются в базу, поэтому прогресс не теряется при перезапуске.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import sqlite3
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from openai import APIError, AsyncOpenAI

from dnd2024_reference import (
    ABILITIES,
    ABILITY_FULL_RU,
    ABILITY_GENITIVE_RU,
    ARMOR,
    CLASSES,
    MAX_LEVEL,
    SHIELD_BONUS,
    SPECIES,
    WEAPONS,
    ability_modifier,
    ability_priority_for_class,
    average_hit_points,
    build_reference_digest,
    class_name_from_text,
    format_modifier,
    hit_die_sides,
    next_xp_threshold,
    normalize_ability_key,
    proficiency_bonus,
    species_name,
    standard_array_for_class,
    starter_equipment_for_class,
    starting_gold_for_class,
)

# ---------------------------------------------------------------------------
# 1. КОНФИГУРАЦИЯ
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# Локальная база данных SQLite с листами персонажей и историей диалогов.
DB_PATH = BASE_DIR / "bot_database.db"

# Загружаем переменные окружения из .env (если файла нет — берём из окружения ОС).
load_dotenv(ENV_PATH)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

MAX_HISTORY_MESSAGES = 20        # сколько последних сообщений держим в памяти и грузим из БД
DM_MAX_TOKENS = 1200             # лимит длины ответа Мастера
DM_TEMPERATURE = 1.15            # чуть выше 1.0 — для более образного и «живого» текста
TELEGRAM_MESSAGE_LIMIT = 4000    # с запасом к лимиту Telegram в 4096 символов
MAX_PLAYER_INPUT = 4000          # обрезаем слишком длинные сообщения игрока
MAX_DICE_COUNT = 100             # защита от /roll 100000d100
MAX_DICE_SIDES = 1000            # защита от /roll d999999

_rng = random.SystemRandom()     # честный ГПСЧ для бросков

# ---------------------------------------------------------------------------
# 2. ЛОГГИРОВАНИЕ
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dnd-dm-bot")
# Логи самого aiogram о каждом апдейте слишком шумные — приглушаем.
logging.getLogger("aiogram.event").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# 3. СИСТЕМНЫЙ ПРОМПТ (DUNGEON MASTER)
# ---------------------------------------------------------------------------

BASE_DM_PROMPT = """
Ты — Мастер Подземелий (Dungeon Master) для настольной ролевой игры Dungeons & Dragons 5e
(редакция 2024 года, также известная как «5.5e»). Ты ведёшь приключение для одного игрока
в личном чате Telegram. Ты одновременно и рассказчик, и судья правил.

# СТИЛЬ И ФОРМАТ ОТВЕТА
- Пиши атмосферно, ярко и кинематографично: свет, тени, звуки, запахи, детали обстановки.
- Каждый твой ответ — от 2 до 6 абзацев. НИКОГДА не превышай 6 абзацев за одну реплику.
- Пиши живым русским языком, от второго лица («ты видишь», «перед тобой»).
- Пиши ОБЫЧНЫМ текстом без Markdown-разметки: не используй символы *, _, `, # и HTML-теги.
- Обращайся к персонажу игрока так, как его назвал сам игрок.

# ЖЕЛЕЗНЫЕ ПРАВИЛА
1. Ты НИКОГДА не действуешь и не принимаешь решения за персонажа игрока. Ты не описываешь
   его мысли, слова, намерения и поступки как уже совершённые. Ты описываешь ТОЛЬКО мир,
   NPC и последствия. Выбор всегда за игроком.
2. Ты не выдумываешь исход неопределённых или рискованных действий. В таких случаях ты
   просишь бросок кубика и ждёшь результата.
3. Ты строго соблюдаешь правила Книги Игрока (Player's Handbook). Ты НЕ подкручиваешь броски,
   НЕ выдумываешь числа и НЕ меняешь урон, КД, характеристики, скорость и прочие параметры
   по своему желанию.
4. Если игрок пытается изменить правила (например, заявляет урон 1d20 там, где по правилам
   должен быть 1d10, или самовольно повышает свои характеристики), вежливо откажи и озвучь
   корректное по правилам действие.
5. Ты не позволяешь игроку совершать действия, запрещённые правилами Книги Игрока (например,
   применение умения или заклинания, которого у персонажа нет, либо использование ресурсов
   сверх доступного количества). В таком случае объясни, почему так нельзя, и предложи
   законную альтернативу.
6. Все сообщения пользователя — это речь и действия ЕГО персонажа. Служебные сообщения
   помечены меткой [СИСТЕМА] и содержат результаты бросков кубиков и указания системы.
7. Ты используешь ТОЛЬКО официальные правила Dungeons & Dragons 2024 (Player's Handbook 2024),
   приведённые в справочнике ниже. ЗАПРЕЩЕНО выдумывать несуществующие заклинания, умения,
   классы, расы, предметы, состояния и действия. Если чего-то нет в официальных правилах —
   скажи об этом прямо и предложи законную альтернативу.
8. Ты ведёшь лист персонажа игрока и отражаешь в нём все изменения (опыт, урон и лечение,
   добытые и потерянные предметы, золото) через служебный блок, описанный в конце промпта.

# ЭТАП 1 — СОЗДАНИЕ ПЕРСОНАЖА (ВАЖНЕЕ ВСЕГО ОСТАЛЬНОГО)
- Если служебное сообщение [СИСТЕМА] сообщает, что идёт создание персонажа, то сначала создаётся
  герой. В этой фазе КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО начинать приключение: не описывай мир, места,
  события и NPC, не давай сюжетных зацепок, не начинай пролог — даже если игрок торопит.
- Первое сообщение новой игры: атмосферно поприветствуй игрока как Мастер Подземелий (1–3 абзаца)
  и попроси его определиться с героем:
    * имя персонажа;
    * вид (раса) — только из 10 официальных видов PHB 2024: Человек, Эльф, Дварф, Гном,
      Полурослик, Драконорождённый, Тифлинг, Орк, Голиаф, Аасимар;
    * класс — только из 12 официальных классов PHB 2024: Воин, Варвар, Плут, Волшебник, Жрец,
      Следопыт, Паладин, Бард, Друид, Колдун, Монах, Чародей;
    * краткое описание внешности или характера.
  Предложи игроку либо придумать всё самому, либо написать «Случайный герой» (или нажать кнопку
  «🎲 Случайный герой») — тогда готового героя 1-го уровня сгенерирует система по правилам
  PHB 2024. Заканчивай реплику вопросом о герое, а не вопросом «Что ты делаешь?».
- Как только игрок назвал данные героя (например, «Меня зовут Боб, я человек воин»), в КОНЦЕ
  этого же ответа добавь служебный JSON-блок с полями "name", "race", "class_name" и, если игрок
  описал внешность или характер, "description".
- Характеристики, максимум и текущие HP, снаряжение и золото НЕ выдумывай и в блок не пиши:
  их по правилам PHB 2024 выставит система (стандартный набор характеристик по классу,
  максимум HP = кость хитов + ТЕЛ, стартовый набор снаряжения 1-го уровня).
- Затем попроси игрока подтвердить героя («Всё верно: Боб, человек-воин?») и жди ответа.
  Просит что-то изменить — исправь поля в служебном блоке и снова попроси подтверждения.
- Только когда система сообщит [СИСТЕМА], что игрок подтвердил героя, начинай вводную сцену
  пролога («Боб, твоя история начинается…») и дальше ведёшь обычное приключение.

# БРОСКИ КУБИКОВ
- В неопределённых, рискованных или опасных ситуациях требуй бросок d20 с модификатором:
  проверку характеристики, спасбросок или бросок атаки.
- Всегда указывай, ЧТО именно нужно проверить, и какой модификатор применить, например:
  «Сделай проверку Ловкости (Акробатика): /roll d20+3» или
  «Брось спасбросок Телосложения: /roll d20+2».
  Модификатор также можно применить кнопкой нужной характеристики (🧠 Проверки по статам).
- Кубики бросает только код бота: игрок жмёт кнопки бросков (🎲 d20, с преимуществом,
  с помехой, урон) либо вводит команду /roll — ты сам броски не выполняешь и результаты
  не придумываешь.
- Когда система сообщает результат броска, опиши исход по правилам D&D 5e: успех, провал,
  частичный результат и их последствия в сцене.

# ЗАВЕРШЕНИЕ ХОДА
- Каждую свою реплику обязательно заканчивай прямым вопросом игроку: «Что ты делаешь?»
  В фазе создания персонажа вместо этого задай вопрос о герое (имя, вид, класс, подтверждение).
""".strip()

# Инструкция о скрытом служебном блоке изменений листа персонажа.
CONTROL_BLOCK_INSTRUCTIONS = """
# СКРЫТЫЙ СЛУЖЕБНЫЙ БЛОК ИЗМЕНЕНИЙ (ЛИСТ ПЕРСОНАЖА)
Ты ведёшь лист персонажа игрока. Всякий раз, когда по ходу сцены состояние персонажа
изменилось (получен опыт, нанесён или вылечен урон, добыты или потеряны предметы,
изменилось золото, или игрок впервые описывает своего героя), в самом КОНЦЕ своего ответа
добавь РОВНО ОДИН служебный JSON-блок внутри тройных обратных кавычек:

```json
{
  "xp_gained": 0,
  "hp_change": 0,
  "add_items": [],
  "remove_items": [],
  "gp_change": 0
}
```

Значение полей:
- "xp_gained" — целое число: опыт за преодолённую опасность, решённую задачу или победу
  (обычно 10–100 за сцену; не выдавай опыт за сам факт боя без результата).
- "hp_change" — целое число: отрицательное при уроне, положительное при лечении.
- "add_items" / "remove_items" — массивы строк: полученные и потерянные предметы и оружие.
- "gp_change" — целое число: изменение количества золота.

Дополнительно, ТОЛЬКО при создании персонажа (когда игрок описывает своего героя), в этом же
блоке укажи строковые поля: "name" (имя), "race" (вид/раса) и "class_name" (класс), а также
"description" — краткое описание внешности или характера, если игрок его дал.

Характеристики ("abilities"), "level", "max_hp", "current_hp", снаряжение и золото при создании
героя НЕ указывай: их по правилам PHB 2024 выставит система. Эти поля (объект "abilities" вида
{"str": 15, "dex": 14, "con": 13, "int": 12, "wis": 10, "cha": 8}; "level"; "max_hp";
"current_hp"; "heroic_inspiration" (true/false)) используй только для последующих изменений
листа по ходу игры.

Правила блока (строго):
- Блок не виден игроку: НЕ упоминай его и не пересказывай в тексте ответа.
- Если изменений нет — блок НЕ добавляй.
- Все числа должны строго соответствовать правилам D&D 2024.
""".strip()

# Полный системный промпт: стиль Мастера + официальный справочник правил + служебный блок.
SYSTEM_PROMPT = "\n\n".join((BASE_DM_PROMPT, build_reference_digest(), CONTROL_BLOCK_INSTRUCTIONS))

# ---------------------------------------------------------------------------
# 4. СТАТИЧНЫЕ ТЕКСТЫ И СЛУЖЕБНЫЕ СООБЩЕНИЯ
# ---------------------------------------------------------------------------

WELCOME_TEXT = (
    "🐉 Добро пожаловать за стол, искатель приключений!\n\n"
    "Я — твой Мастер Подземелий в духе Dungeons & Dragons 5e (2024). Я опишу мир, его "
    "опасности и судьбу твоего героя, но все решения остаются за тобой.\n\n"
    "Шаг 1 — создай героя:\n"
    "• Назови имя, вид (раса), класс и пару слов о внешности или характере героя.\n"
    "• Виды по правилам PHB 2024: Человек, Эльф, Дварф, Гном, Полурослик, Драконорождённый, "
    "Тифлинг, Орк, Голиаф, Аасимар.\n"
    "• Классы: Воин, Варвар, Плут, Волшебник, Жрец, Следопыт, Паладин, Бард, Друид, Колдун, "
    "Монах, Чародей.\n"
    "• Не хочешь придумывать сам — напиши «Случайный герой» или нажми кнопку "
    "«🎲 Случайный герой»: я соберу героя 1-го уровня строго по правилам PHB 2024 "
    "(характеристики, HP и стартовое снаряжение считает код бота).\n"
    "• Затем подтверди героя («да» или кнопка «✅ Подтвердить героя») — и начнётся пролог.\n\n"
    "Как играть:\n"
    "• Пиши обычными сообщениями, что делает и говорит твой персонаж.\n"
    "• Когда исход поступка неочевиден или опасен, я попрошу бросок кубика.\n"
    "• Броски делает только код бота: жми кнопки 🎲 d20, 🎲 d20 с преим./помех., 🎲 Бросок урона "
    "или используй команду /roll, например /roll d20 или /roll 2d6+3.\n"
    "• Проверки характеристик с модификатором — кнопка 🧠 Проверки по статам или команда /check.\n"
    "• Играй по правилам Книги Игрока 2024 — я не приму накрученные броски и урон не по правилам.\n"
    "• Веди лист персонажа: кнопки «📜 Лист» и «🎒 Инвентарь», команды /sheet и /inventory.\n"
    "Команды: /start, /reset, /hero, /roll <кубик>, /sheet, /inventory, /check.\n\n"
    "Предыдущая сессия сброшена. Сперва — герой, потом — приключение!"
)

# ---------------------------------------------------------------------------
# ЭТАП 1: ТЕКСТЫ СОЗДАНИЯ ПЕРСОНАЖА
# ---------------------------------------------------------------------------

# Фазы общения с игроком (см. creation_stage).
CREATION_STAGE_HERO = "creation"          # герой ещё не описан
CREATION_STAGE_CONFIRM = "confirmation"   # герой описан, ждём подтверждения игрока
CREATION_STAGE_PLAY = "play"              # герой подтверждён, идёт приключение

# Инструкция Мастеру для самого первого сообщения новой игры: только приветствие и герой.
HERO_CREATION_START_PROMPT = (
    "[СИСТЕМА] Новая игра. История диалога пуста, лист персонажа пуст: имя, вид (раса) и класс "
    "ещё не заданы. Идёт ЭТАП 1 — создание персонажа, приключение ещё НЕ началось.\n"
    "Задание: атмосферно поприветствуй игрока как Мастер Подземелий (1–3 абзаца) и попроси его "
    "определиться с героем: имя; вид (раса) — только из 10 официальных видов PHB 2024 (Человек, "
    "Эльф, Дварф, Гном, Полурослик, Драконорождённый, Тифлинг, Орк, Голиаф, Аасимар); класс — "
    "только из 12 официальных классов PHB 2024 (Воин, Варвар, Плут, Волшебник, Жрец, Следопыт, "
    "Паладин, Бард, Друид, Колдун, Монах, Чародей); краткое описание внешности или характера.\n"
    "Предложи игроку либо придумать героя самому, либо написать «Случайный герой» / нажать кнопку "
    "«🎲 Случайный герой» — тогда героя сгенерирует система по правилам PHB 2024.\n"
    "НЕ описывай мир, сцену, события и зацепки, не начинай пролог — только создание героя. "
    "Закончи реплику вопросом о герое (не «Что ты делаешь?»)."
)

# Инструкция Мастеру для продолжения фазы создания (герой всё ещё не описан).
HERO_CREATION_PROMPT = (
    "[СИСТЕМА] Продолжается ЭТАП 1 — создание персонажа: в листе всё ещё нет имени, вида или "
    "класса. Приключение НЕ началось: не описывай мир, сцену, события и зацепки, не начинай "
    "пролог, даже если игрок просит начать игру.\n"
    "Задание: ответь игроку по существу и помоги завершить героя. Как только он назовёт данные "
    "героя, добавь в КОНЕЦ ответа служебный JSON-блок с полями \"name\", \"race\", \"class_name\" "
    "и (если игрок описал внешность или характер) \"description\". Характеристики, HP, снаряжение "
    "и золото не выдумывай — их выставит система по правилам PHB 2024. Затем попроси игрока "
    "подтвердить героя."
)

# Тексты, которые бот показывает игроку после создания героя.
HERO_CARD_TITLE_CREATED = "✅ Герой записан в лист персонажа!"
HERO_CARD_TITLE_RANDOM = "🎲 Случайный герой готов!"
HERO_CARD_TITLE_PENDING = "📜 Герой ещё не подтверждён — проверь его перед началом игры"

HERO_CARD_QUESTION = (
    "Всё верно? Напиши «да» или нажми «✅ Подтвердить героя» — и начнётся пролог.\n"
    "Хочешь что-то поменять (имя, вид, класс, внешность) — просто напиши, что изменить."
)

HERO_NOT_CREATED_ALERT = (
    "Сначала закончим героя: нужны имя, вид (раса) и класс.\n"
    "Опиши их в сообщении или нажми «🎲 Случайный герой»."
)

HERO_ALREADY_CONFIRMED_ALERT = "Герой уже подтверждён — приключение идёт. Что ты делаешь?"

HERO_ALREADY_CONFIRMED_TEXT = (
    "✅ Герой уже подтверждён, приключение идёт полным ходом.\n\nЧто ты делаешь?"
)

# Признаки того, что игрок просит сгенерировать героя вместо описания своего.
RANDOM_HERO_MARKERS: tuple[str, ...] = (
    "случайн", "наугад", "рандом", "random", "любой герой", "выбери за меня",
    "сгенерируй геро", "сгенерируй персон", "придумай за меня", "составь за меня",
)

# Согласие игрока подтвердить героя: короткое сообщение вида «да», «подтверждаю», «начинаем».
HERO_CONFIRM_PATTERN = re.compile(
    r"^\s*(?:я\s+)?(?:да|ага|угу|верно|всё верно|все верно|всё правильно|все правильно|"
    r"подтверждаю(?:\s+героя)?|согласен|согласна|ок|окей|окэй|хорошо|принято|готов|готова|"
    r"начинаем|начинай|поехали|играем|давай|yes|yep|ok|go)\s*,?\s*"
    r"(?:начинаем|начинай|поехали|играем|играть|давай|в путь|герой|героем|этим героем)?"
    r"\s*[!.,…]*\s*$",
    re.IGNORECASE,
)

# Имя героя из явного представления: «Меня зовут Боб», «зови меня Грим».
HERO_NAME_PATTERN = re.compile(
    r"(?:меня зовут|моё имя|мое имя|зови меня|зовут меня)\s+"
    r"(?P<name>[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё'\-]{1,31})",
    re.IGNORECASE,
)

# Имена и заготовки характера для случайного героя: (имя, описание внешности/характера).
RANDOM_HERO_PROFILES: tuple[tuple[str, str], ...] = (
    ("Каэлен", "Сухощавый, с обветренным лицом и внимательным взглядом; молчалив, но упрям."),
    ("Брунхильда", "Крепкая, с косой цвета соломы; смеётся громко, а в драке не отступает."),
    ("Тэм", "Невысокий ловкач с быстрыми глазами и вечной ухмылкой; ценит тишину и монету."),
    ("Селена", "Стройная, с серебристой прядью в волосах; говорит спокойно и смотрит прямо."),
    ("Грим", "Широкоплечий бородач со шрамами на руках; грубоват, но верен слову."),
    ("Нисса", "Хрупкая на вид, с цепкими пальцами и живым умом; любопытна до неприличия."),
    ("Ортега", "Загорелый, с кольцами в бороде; в пути поёт, чтобы не думать о прошлом."),
    ("Виллемина", "Молодая, с медной кожей и упрямым подбородком; верит, что удача — это умение."),
    ("Драган", "Высокий, с тяжёлым взглядом и спокойными движениями; сперва думает, потом бьёт."),
    ("Лиора", "Светловолосая, в дорожном плаще; собирает чужие истории, свою пока не рассказала."),
)


ROLL_USAGE_TEXT = (
    "Формат броска: /roll <кубик>\n"
    "Примеры: /roll d20, /roll 2d6+3, /roll 1d10-1, /roll 4d6"
)

# Тексты инлайн-клавиатур и «всплывающих» подсказок на кнопках.
ACTION_MENU_TEXT = "🎲 Быстрые действия: выбери бросок или нужный раздел."

CHECKS_MENU_TEXT = (
    "🧠 ПРОВЕРКИ ХАРАКТЕРИСТИК\n"
    "Нажми характеристику — я брошу d20, добавлю её модификатор из твоего листа "
    "и передам результат Мастеру."
)

STALE_CALLBACK_TEXT = (
    "Эта кнопка устарела (сообщение слишком старое). "
    "Напиши новое действие в чат — под свежим ответом кнопки снова активны."
)

UNKNOWN_BUTTON_TEXT = "Неизвестная кнопка — попробуй ещё раз."

# Короткая инструкция для Мастера после любого броска, сделанного кодом бота.
DM_ROLL_INSTRUCTION = (
    "Опиши исход этого броска в рамках текущей сцены по правилам D&D 5e "
    "(успех, провал и его последствия). Закончи ход вопросом «Что ты делаешь?»"
)

API_ERROR_TEXT = (
    "⚠️ Мастер ненадолго отвлёкся: не удалось связаться с оракулом (ошибка DeepSeek API).\n"
    "Попробуй повторить сообщение через несколько секунд."
)

GENERIC_ERROR_TEXT = (
    "⚠️ Что-то пошло не так при обработке твоего действия. Попробуй ещё раз."
)

# ---------------------------------------------------------------------------
# 5. КУБИКИ (парсинг и броски считает код, а не нейросеть)
# ---------------------------------------------------------------------------

# Поддерживаем латинскую «d» и кириллическую «д», а также необязательные пробелы.
DICE_PATTERN = re.compile(
    r"^\s*(?P<count>\d{0,3})\s*[dDдД]\s*(?P<sides>\d{1,4})\s*(?P<modifier>[+-]\s*\d{1,4})?\s*$"
)


@dataclass(slots=True)
class DiceRoll:
    """Результат броска кубиков в конкретной нотации.

    ``mode`` («advantage»/«disadvantage») используется только для d20:
    бросаются два кубика, а в зачёт идёт лучший/худший (см. ``kept``).
    Пустой список ``rolls`` означает фиксированное значение без броска кубиков.
    """

    count: int
    sides: int
    rolls: list[int] = field(default_factory=list)
    modifier: int = 0
    mode: str = ""

    @classmethod
    def flat(cls, amount: int) -> "DiceRoll":
        """Бросок без кубиков: фиксированное значение (например, урон «1»)."""
        return cls(count=0, sides=0, rolls=[], modifier=_as_int(amount))

    @property
    def is_flat(self) -> bool:
        """True, если кубики не бросались (только фиксированный модификатор)."""
        return not self.rolls

    @property
    def notation(self) -> str:
        """Нотация броска, например «2d6+3»."""
        if self.is_flat:
            return format_modifier(self.modifier)
        notation = f"{self.count}d{self.sides}"
        if self.modifier:
            notation += f"{self.modifier:+d}"
        return notation

    @property
    def mode_label(self) -> str:
        """Подпись варианта броска d20: преимущество, помеха или пусто."""
        if self.mode == "advantage":
            return " (с преимуществом)"
        if self.mode == "disadvantage":
            return " (с помехой)"
        return ""

    @property
    def kept(self) -> list[int]:
        """Кубики, которые идут в зачёт (при преимуществе/помехе — один из двух d20)."""
        if self.mode == "advantage" and len(self.rolls) > 1:
            return [max(self.rolls)]
        if self.mode == "disadvantage" and len(self.rolls) > 1:
            return [min(self.rolls)]
        return list(self.rolls)

    @property
    def total(self) -> int:
        """Итоговая сумма броска с учётом модификатора."""
        return sum(self.kept) + self.modifier

    def describe(self) -> str:
        """Человекочитаемая «математика» броска для игрока."""
        if self.is_flat:
            return f"фиксированный урон {self.total}"
        if self.mode and len(self.rolls) > 1:
            # Преимущество/помеха: показываем оба кубика и тот, что пошёл в зачёт.
            body = f"{self.rolls[0]}, {self.rolls[1]} → в зачёт {self.kept[0]}"
        else:
            body = " + ".join(str(value) for value in self.rolls)
        if self.modifier > 0:
            body += f" + {self.modifier}"
        elif self.modifier < 0:
            body += f" - {abs(self.modifier)}"
        return f"{self.notation}{self.mode_label}: {body} = {self.total}"

    def context_message(self, purpose: str = "") -> str:
        """Служебное сообщение о броске для контекста Мастера.

        :param purpose: зачем бросали (например, «бросок урона: Длинный меч (1d8)»).
        """
        what = purpose or "бросок кубиков"
        if self.is_flat:
            detail = f"Бросок без кубиков, фиксированное значение: {self.total}"
        else:
            dice = ", ".join(map(str, self.rolls))
            if self.mode and len(self.rolls) > 1:
                dice += f" (в зачёт {self.kept[0]})"
            modifier_text = f"{self.modifier:+d}" if self.modifier else "без модификатора"
            detail = (
                f"Нотация: {self.notation}{self.mode_label}. Выпало: {dice} "
                f"({modifier_text}). Итог: {self.total}"
            )
        return f"[СИСТЕМА] Игрок сделал {what}. {detail}. {DM_ROLL_INSTRUCTION}"


def make_roll(count: int, sides: int, modifier: int = 0, mode: str = "") -> DiceRoll:
    """Бросает кубики кодом бота (кнопки, проверки, урон). Значения защищены от «мусора».

    :param count: сколько кубиков бросать (1..MAX_DICE_COUNT).
    :param sides: число граней кубика (2..MAX_DICE_SIDES).
    :param modifier: модификатор к сумме (может быть отрицательным).
    :param mode: «advantage»/«disadvantage» для d20 — в зачёт идёт лучший/худший кубик.
    """
    count = max(1, min(_as_int(count), MAX_DICE_COUNT))
    sides = max(2, min(_as_int(sides), MAX_DICE_SIDES))
    rolls = [_rng.randint(1, sides) for _ in range(count)]
    return DiceRoll(count=count, sides=sides, rolls=rolls, modifier=_as_int(modifier), mode=mode)


def parse_and_roll(expression: str) -> DiceRoll:
    """
    Разбирает нотацию кубиков и выполняет бросок.

    :raise ValueError: если нотация некорректна или выходит за допустимые пределы.
    """
    match = DICE_PATTERN.match(expression)
    if match is None:
        raise ValueError("Неверный формат броска.")

    count = int(match.group("count") or 1)
    sides = int(match.group("sides"))
    raw_modifier = match.group("modifier")
    modifier = int(raw_modifier.replace(" ", "")) if raw_modifier else 0

    if not 1 <= count <= MAX_DICE_COUNT:
        raise ValueError(f"Число кубиков должно быть от 1 до {MAX_DICE_COUNT}.")
    if not 2 <= sides <= MAX_DICE_SIDES:
        raise ValueError(f"Число граней кубика должно быть от 2 до {MAX_DICE_SIDES}.")

    return make_roll(count=count, sides=sides, modifier=modifier)


# ---------------------------------------------------------------------------
# 6. ЛИСТ ПЕРСОНАЖА (Character Sheet)
# ---------------------------------------------------------------------------

# Значения характеристик по умолчанию: 10 — «средний» герой без распределённых очков.
DEFAULT_ABILITIES: dict[str, int] = {code: 10 for code in ABILITIES}

# Заглушки для ещё не созданного героя.
DEFAULT_NAME = "Безымянный герой"
DEFAULT_RACE = "Не определена"
DEFAULT_CLASS = "Не определён"

# Границы значений характеристик по правилам D&D.
MIN_ABILITY, MAX_ABILITY = 1, 30

# Особое значение current_hp «ещё не задано»: при создании HP = максимум.
# Нужно, чтобы честный 0 HP (персонаж без сознания), загруженный из базы,
# не превращался обратно в полное здоровье.
UNSET_HP = -1


def _as_int(value: Any) -> int:
    """Аккуратно приводит значение из «сырого» JSON модели к int (0 при неудаче)."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return 0
    return 0


def _as_text_list(value: Any) -> list[str]:
    """Приводит значение к списку непустых строк (для add_items / remove_items)."""
    if isinstance(value, str):
        candidates: list[Any] = [value]
    elif isinstance(value, (list, tuple)):
        candidates = list(value)
    else:
        return []

    items: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            candidate = str(candidate)
        if isinstance(candidate, str) and candidate.strip():
            items.append(candidate.strip()[:80])
    return items


def _remove_first(items: list[str], target: str) -> bool:
    """Удаляет первое совпадение по названию (без учёта регистра). True при успехе."""
    needle = target.strip().lower()
    for index, item in enumerate(items):
        if item.lower() == needle:
            del items[index]
            return True
    return False


def _as_optional_text(value: Any) -> Optional[str]:
    """Возвращает непустую строку или None (для загрузки полей из базы)."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


@dataclass
class Character:
    """
    Лист персонажа игрока (D&D 2024).

    Хранит всё состояние героя: имя, расу, класс, уровень, HP, опыт (XP),
    характеристики, инвентарь и золото. Развивается автоматически по опыту через
    стандартные пороги уровней D&D 2024 (см. dnd2024_reference).
    """

    name: str = DEFAULT_NAME
    race: str = DEFAULT_RACE
    class_name: str = DEFAULT_CLASS
    level: int = 1
    current_hp: int = UNSET_HP
    max_hp: int = 0
    xp: int = 0
    gp: int = 0
    abilities: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_ABILITIES))
    inventory: list[str] = field(default_factory=list)
    heroic_inspiration: bool = False
    description: str = ""
    # Признак «игрок подтвердил героя». Пока он False, идёт этап создания персонажа
    # (см. creation_stage) и приключение с прологом не начинается.
    hero_confirmed: bool = False

    def __post_init__(self) -> None:
        """Нормализует «сырые» данные: обрезает строки и приводит числа к правилам."""
        self.name = (self.name or "").strip()[:64] or DEFAULT_NAME
        self.race = (self.race or "").strip()[:64] or DEFAULT_RACE
        self.class_name = (self.class_name or "").strip()[:64] or DEFAULT_CLASS
        self.level = max(1, min(_as_int(self.level) or 1, MAX_LEVEL))

        normalized: dict[str, int] = {}
        for code in ABILITIES:
            value = _as_int(self.abilities.get(code, 10))
            normalized[code] = max(MIN_ABILITY, min(value, MAX_ABILITY))
        self.abilities = normalized

        self.xp = max(0, _as_int(self.xp))
        self.gp = max(0, _as_int(self.gp))

        if self.max_hp <= 0:
            # HP 1-го уровня = максимум кости хитов + модификатор ТЕЛ.
            self.max_hp = max(1, self.hit_die + self.ability_mod("con"))
        if self.current_hp < 0:
            self.current_hp = self.max_hp
        else:
            self.current_hp = max(0, min(_as_int(self.current_hp), self.max_hp))
        self.inventory = _as_text_list(self.inventory)
        self.description = (self.description or "").strip()[:400]

    # --- Проверки состояния персонажа ---

    @property
    def is_created(self) -> bool:
        """True, если игрок уже описал героя: заданы имя, вид (раса) и класс."""
        return (
            self.name != DEFAULT_NAME
            and self.race != DEFAULT_RACE
            and self.class_name != DEFAULT_CLASS
        )

    # --- Производные характеристики (считаются по правилам) ---

    def ability_mod(self, code: str) -> int:
        """Модификатор характеристики по её коду ('str', 'dex', ...)."""
        return ability_modifier(self.abilities.get(code, 10))

    @property
    def hit_die(self) -> int:
        """Число граней кости хитов класса (d6/d8/d10/d12)."""
        return hit_die_sides(self.class_name)

    @property
    def proficiency_bonus(self) -> int:
        """Бонус мастерства по текущему уровню."""
        return proficiency_bonus(self.level)

    @property
    def initiative(self) -> int:
        """Модификатор инициативы (равен модификатору ЛОВ)."""
        return self.ability_mod("dex")

    @property
    def armor_class(self) -> int:
        """КД по правилам PHB 2024: доспех из снаряжения + модификатор ЛОВ (+2 за щит)."""
        armor = self.best_armor
        dex_mod = self.ability_mod("dex")

        if armor is None:
            armor_class = 10 + dex_mod
        else:
            _name, base_ac, max_dex = armor
            if max_dex == 0:            # тяжёлый доспех — модификатор ЛОВ не добавляется
                armor_class = base_ac
            elif max_dex is None:       # лёгкий доспех — модификатор ЛОВ без ограничения
                armor_class = base_ac + dex_mod
            else:                       # средний доспех — модификатор ЛОВ не выше предела
                armor_class = base_ac + min(dex_mod, max_dex)

        if self.has_shield:
            armor_class += SHIELD_BONUS
        return armor_class

    @property
    def best_armor(self) -> Optional[tuple[str, int, Optional[int]]]:
        """Лучший доспех из снаряжения: (название, база КД, предел модификатора ЛОВ)."""
        best: Optional[tuple[str, int, Optional[int]]] = None
        for item in self.inventory:
            lowered = item.strip().lower()
            for name, _category, base_ac, max_dex, _strength, _stealth in ARMOR:
                if name.lower() in lowered and (best is None or base_ac > best[1]):
                    best = (name, base_ac, max_dex)
        return best

    @property
    def has_shield(self) -> bool:
        """Есть ли щит в снаряжении героя."""
        return any("щит" in item.lower() for item in self.inventory)

    @property
    def armor_label(self) -> str:
        """Подпись доспеха для листа персонажа: «Кольчуга, щит» или «без доспехов»."""
        names: list[str] = []
        if self.best_armor is not None:
            names.append(self.best_armor[0])
        if self.has_shield:
            names.append("щит")
        return ", ".join(names) if names else "без доспехов"

    @property
    def passive_perception(self) -> int:
        """Пассивная Внимательность: 10 + модификатор МУД."""
        return 10 + self.ability_mod("wis")

    @property
    def is_alive(self) -> bool:
        """Жив ли персонаж (HP выше нуля)."""
        return self.current_hp > 0

    # --- Изменение листа персонажа ---

    def _level_up(self) -> str:
        """Повышает уровень: увеличивает максимум HP и лечит на ту же величину."""
        self.level += 1
        gained = average_hit_points(self.hit_die, self.ability_mod("con"))
        self.max_hp += gained
        self.current_hp = min(self.max_hp, self.current_hp + gained)
        return (
            f"⬆️ НОВЫЙ УРОВЕНЬ: {self.level}! Максимум HP: {self.max_hp} (+{gained}). "
            f"Бонус мастерства: {format_modifier(self.proficiency_bonus)}."
        )

    def _apply_level_ups(self) -> list[str]:
        """Повышает уровень столько раз, сколько позволяет накопленный опыт."""
        notes: list[str] = []
        while self.level < MAX_LEVEL:
            threshold = next_xp_threshold(self.level)
            if threshold is None or self.xp < threshold:
                break
            notes.append(self._level_up())
        return notes

    def _apply_optional_sheet(self, data: dict[str, Any]) -> list[str]:
        """Применяет необязательные поля листа (создание или правка персонажа Мастером)."""
        notes: list[str] = []

        name = data.get("name")
        if isinstance(name, str) and name.strip() and name.strip() != self.name:
            self.name = name.strip()[:64]
            notes.append(f"📛 Имя персонажа: {self.name}.")

        race = data.get("race")
        if isinstance(race, str) and race.strip() and race.strip() != self.race:
            self.race = race.strip()[:64]
            notes.append(f"🧬 Раса: {self.race}.")

        class_name = data.get("class_name")
        if isinstance(class_name, str) and class_name.strip():
            cleaned = class_name.strip()[:64]
            if cleaned != self.class_name:
                self.class_name = cleaned
                notes.append(f"⚔️ Класс: {self.class_name} (кость хитов d{self.hit_die}).")

        description = data.get("description")
        if isinstance(description, str) and description.strip():
            cleaned_description = description.strip()[:400]
            if cleaned_description != self.description:
                self.description = cleaned_description
                notes.append("🖋️ Описание героя записано.")

        level = _as_int(data.get("level"))
        if 1 <= level <= MAX_LEVEL and level != self.level:
            self.level = level
            notes.append(f"🎖️ Уровень: {self.level}.")

        abilities = data.get("abilities")
        if isinstance(abilities, dict):
            changed: list[str] = []
            for raw_key, raw_value in abilities.items():
                code = normalize_ability_key(raw_key)
                if code is None:
                    continue
                value = max(MIN_ABILITY, min(_as_int(raw_value), MAX_ABILITY))
                if value != self.abilities[code]:
                    self.abilities[code] = value
                    changed.append(f"{ABILITIES[code]} {value}")
            if changed:
                notes.append("🧠 Характеристики: " + ", ".join(changed) + ".")

        max_hp = _as_int(data.get("max_hp"))
        if max_hp > 0:
            self.max_hp = max_hp
            self.current_hp = min(self.current_hp, self.max_hp)

        if data.get("current_hp") is not None:
            self.current_hp = max(0, min(_as_int(data.get("current_hp")), self.max_hp))

        inspiration = data.get("heroic_inspiration")
        if isinstance(inspiration, bool):
            self.heroic_inspiration = inspiration
            if inspiration:
                notes.append("🌟 Вдохновение героя получено!")

        return notes

    def apply_control(self, data: dict[str, Any]) -> list[str]:
        """
        Применяет служебный JSON-блок Мастера к листу персонажа.

        Возвращает список уведомлений для игрока (опыт, урон, предметы, золото, уровень).
        """
        notes: list[str] = list(self._apply_optional_sheet(data))

        xp_gained = _as_int(data.get("xp_gained"))
        if xp_gained > 0:
            self.xp += xp_gained
            notes.append(f"✨ Получено {xp_gained} XP (всего {self.xp}).")
            notes.extend(self._apply_level_ups())

        hp_change = _as_int(data.get("hp_change"))
        if hp_change:
            before = self.current_hp
            self.current_hp = max(0, min(self.current_hp + hp_change, self.max_hp))
            delta = self.current_hp - before
            if delta < 0:
                notes.append(f"💔 Потеряно {-delta} HP (осталось {self.current_hp}/{self.max_hp}).")
            elif delta > 0:
                notes.append(f"💚 Восстановлено {delta} HP (теперь {self.current_hp}/{self.max_hp}).")
            if self.current_hp == 0:
                notes.append("☠️ Персонаж без сознания (0 HP) — нужны спасброски от смерти.")

        for item in _as_text_list(data.get("add_items")):
            self.inventory.append(item)
            notes.append(f"🎁 Получено: {item}.")

        for item in _as_text_list(data.get("remove_items")):
            if _remove_first(self.inventory, item):
                notes.append(f"➖ Потеряно: {item}.")

        gp_change = _as_int(data.get("gp_change"))
        if gp_change:
            self.gp = max(0, self.gp + gp_change)
            if gp_change > 0:
                notes.append(f"💰 +{gp_change} золота (всего {self.gp} gp).")
            else:
                notes.append(f"💸 Потрачено {-gp_change} золота (осталось {self.gp} gp).")

        return notes

    # --- Сериализация для постоянного хранения (SQLite) ---

    def to_dict(self) -> dict[str, Any]:
        """Сериализует лист персонажа в JSON-совместимый словарь."""
        return {
            "name": self.name,
            "race": self.race,
            "class_name": self.class_name,
            "level": self.level,
            "current_hp": self.current_hp,
            "max_hp": self.max_hp,
            "xp": self.xp,
            "gp": self.gp,
            "abilities": {code: int(self.abilities[code]) for code in ABILITIES},
            "inventory": list(self.inventory),
            "heroic_inspiration": bool(self.heroic_inspiration),
            "description": self.description,
            "hero_confirmed": bool(self.hero_confirmed),
        }

    def to_json(self) -> str:
        """Сериализует лист персонажа в строку JSON (с русскими буквами как есть)."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Any) -> "Character":
        """
        Восстанавливает лист персонажа из словаря.

        Терпима к «мусору»: отсутствующие поля берутся по умолчанию, лишние
        игнорируются, а испорченные значения приводятся к допустимым
        (характеристики, уровень, HP, инвентарь).
        """
        if not isinstance(data, Mapping):
            return cls()

        abilities: dict[str, int] = {}
        raw_abilities = data.get("abilities")
        if isinstance(raw_abilities, Mapping):
            for raw_key, raw_value in raw_abilities.items():
                code = normalize_ability_key(raw_key)
                if code is None and raw_key in ABILITIES:
                    code = raw_key
                if code is not None:
                    abilities[code] = _as_int(raw_value)

        inspiration = data.get("heroic_inspiration")
        # Отсутствие current_hp означает «не задано» (полное здоровье), а не 0 HP.
        current_hp = UNSET_HP if data.get("current_hp") is None else _as_int(data.get("current_hp"))

        name = _as_optional_text(data.get("name")) or DEFAULT_NAME
        race = _as_optional_text(data.get("race")) or DEFAULT_RACE
        class_name = _as_optional_text(data.get("class_name")) or DEFAULT_CLASS

        # Записи базы прежних версий не знали про подтверждение героя: если герой уже
        # был создан, считаем его подтверждённым, иначе игрок застрял бы на подтверждении.
        raw_confirmed = data.get("hero_confirmed")
        if isinstance(raw_confirmed, bool):
            hero_confirmed = raw_confirmed
        else:
            hero_confirmed = (
                name != DEFAULT_NAME and race != DEFAULT_RACE and class_name != DEFAULT_CLASS
            )

        return cls(
            name=name,
            race=race,
            class_name=class_name,
            level=_as_int(data.get("level")) or 1,
            current_hp=current_hp,
            max_hp=_as_int(data.get("max_hp")),
            xp=_as_int(data.get("xp")),
            gp=_as_int(data.get("gp")),
            abilities=abilities or dict(DEFAULT_ABILITIES),
            inventory=_as_text_list(data.get("inventory")),
            heroic_inspiration=inspiration if isinstance(inspiration, bool) else False,
            description=_as_optional_text(data.get("description")) or "",
            hero_confirmed=hero_confirmed,
        )

    @classmethod
    def from_json(cls, raw: Any) -> "Character":
        """Восстанавливает лист персонажа из JSON-строки (при ошибке — новый герой)."""
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        if not isinstance(raw, str):
            return cls()
        try:
            data = json.loads(raw)
        except ValueError:
            logger.warning("Повреждённая запись листа персонажа в базе — создаю нового героя.")
            return cls()
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Форматирование листа персонажа и разбор служебного блока Мастера
# ---------------------------------------------------------------------------


def hp_progress_bar(current_hp: int, max_hp: int, width: int = 20) -> str:
    """Рисует прогресс-бар HP, например: ████████░░░░░░░░░░░░"""
    ratio = 0.0 if max_hp <= 0 else max(0.0, min(1.0, current_hp / max_hp))
    filled = round(ratio * width)
    return "█" * filled + "░" * (width - filled)


def format_character_sheet(character: Character) -> str:
    """Форматирует лист персонажа в аккуратное сообщение для игрока."""
    threshold = next_xp_threshold(character.level)
    if threshold is None:
        xp_line = f"Опыт: {character.xp} XP (достигнут максимальный уровень)"
    else:
        xp_line = f"Опыт: {character.xp} / {threshold} XP до {character.level + 1} ур."

    hp_bar = hp_progress_bar(character.current_hp, character.max_hp)
    abilities = [
        f"{ABILITIES[code]} {character.abilities[code]:>2} "
        f"({format_modifier(character.ability_mod(code))})"
        for code in ABILITIES
    ]
    ability_rows = ["   ".join(abilities[index:index + 3]) for index in range(0, 6, 3)]

    if character.inventory:
        inventory = "\n".join(f" • {item}" for item in character.inventory)
        inventory_title = f"🎒 Снаряжение ({len(character.inventory)}):"
    else:
        inventory = " • (пусто)"
        inventory_title = "🎒 Снаряжение:"

    status = "жив" if character.is_alive else "без сознания"
    inspiration = "да" if character.heroic_inspiration else "нет"

    # Описание внешности и характера показываем, только если игрок его задал.
    description_lines = (
        [f"🖋️ Описание: {character.description}"] if character.description else []
    )

    return "\n".join(
        [
            "📜 ЛИСТ ПЕРСОНАЖА",
            "",
            f"📛 Имя: {character.name}",
            f"🧬 Раса: {character.race}",
            f"⚔️ Класс: {character.class_name} (кость хитов d{character.hit_die})",
            *description_lines,
            f"🎖️ Уровень: {character.level} "
            f"(бонус мастерства {format_modifier(character.proficiency_bonus)})",
            f"🧭 Инициатива {format_modifier(character.initiative)} | "
            f"КД {character.armor_class} ({character.armor_label}) | "
            f"Пассивная Внимательность {character.passive_perception}",
            "",
            f"❤️ HP [{hp_bar}] {character.current_hp}/{character.max_hp} ({status})",
            f"🌟 Вдохновение героя: {inspiration}",
            f"💰 Золото: {character.gp} gp",
            "",
            "📊 Прогресс",
            xp_line,
            "",
            "🧠 Характеристики",
            *ability_rows,
            "",
            inventory_title,
            inventory,
        ]
    )


def format_inventory(character: Character) -> str:
    """Компактный список снаряжения и золота для кнопки «🎒 Инвентарь»."""
    if character.inventory:
        title = f"🎒 СНАРЯЖЕНИЕ ({len(character.inventory)}):"
        items = [f" • {item}" for item in character.inventory]
    else:
        title = "🎒 СНАРЯЖЕНИЕ:"
        items = [" • (пусто)"]

    return "\n".join(
        [
            title,
            *items,
            "",
            f"💰 Золото: {character.gp} gp",
            f"⚔️ Класс: {character.class_name} | ❤️ HP: "
            f"{character.current_hp}/{character.max_hp}",
        ]
    )


# ---------------------------------------------------------------------------
# ЭТАП 1: ЛОГИКА СОЗДАНИЯ ПЕРСОНАЖА (героя собирает код, а не нейросеть)
# ---------------------------------------------------------------------------


def creation_stage(character: Character) -> str:
    """Фаза общения с игроком: создание героя, подтверждение героя или сама игра."""
    if not character.is_created:
        return CREATION_STAGE_HERO
    if not character.hero_confirmed:
        return CREATION_STAGE_CONFIRM
    return CREATION_STAGE_PLAY


def hero_summary(character: Character) -> str:
    """Однострочная выжимка о герое для служебных сообщений Мастеру."""
    if not character.is_created:
        missing = [
            title
            for title, value, default in (
                ("имя", character.name, DEFAULT_NAME),
                ("вид (раса)", character.race, DEFAULT_RACE),
                ("класс", character.class_name, DEFAULT_CLASS),
            )
            if value == default
        ]
        return "Лист персонажа ещё не заполнен: не заданы " + ", ".join(missing) + "."

    abilities = ", ".join(
        f"{ABILITIES[code]} {character.abilities[code]}" for code in ABILITIES
    )
    parts = [
        f"имя: {character.name}",
        f"вид (раса): {character.race}",
        f"класс: {character.class_name}",
        f"уровень: {character.level}",
        f"HP: {character.current_hp}/{character.max_hp}",
        f"характеристики: {abilities}",
    ]
    if character.description:
        parts.append(f"описание: {character.description}")
    if character.inventory:
        parts.append("снаряжение: " + ", ".join(character.inventory))
    return "Данные героя — " + "; ".join(parts) + "."


def creation_prompt(session: Session) -> str:
    """Инструкция Мастеру в фазе создания: первое сообщение игры или продолжение."""
    if len(session.history) <= 1:
        return HERO_CREATION_START_PROMPT
    return HERO_CREATION_PROMPT


def hero_confirmation_prompt(character: Character) -> str:
    """Инструкция Мастеру: герой создан, но игрок его ещё не подтвердил."""
    return (
        "[СИСТЕМА] Этап создания персонажа: герой записан в лист, но игрок его ещё НЕ подтвердил. "
        "Приключение и пролог начинать ЗАПРЕЩЕНО.\n"
        f"{hero_summary(character)}\n"
        "Задание: коротко (1–2 абзаца) отреагируй на сообщение игрока. Просит изменить героя — "
        "исправь нужные поля (\"name\", \"race\", \"class_name\", \"description\") в служебном "
        "JSON-блоке; описывает действия вместо героя — вежливо напомни, что сначала нужно "
        "подтвердить героя. В конце спроси, всё ли верно с героем: подтвердить можно словом «да» "
        "или кнопкой «✅ Подтвердить героя». Пролог не начинай."
    )


def prologue_prompt(character: Character) -> str:
    """Инструкция Мастеру начать вводную сцену после подтверждения героя."""
    return (
        "[СИСТЕМА] Игрок подтвердил героя. Этап создания персонажа завершён — начинается игра.\n"
        f"{hero_summary(character)}\n"
        "Задание: начни вводную сцену пролога по правилам D&D 2024: атмосферно опиши, где и как "
        f"начинается путь героя, назови его по имени («{character.name}, твоя история "
        "начинается…»), придумай название мира и короткую завязку в духе тёмного героического "
        "фэнтези, дай одну-две зацепки и остановись в точке выбора. НЕ описывай действия, слова и "
        "мысли героя игрока. Закончи вопросом «Что ты делаешь?»"
    )


def random_hero_prompt(character: Character) -> str:
    """Инструкция Мастеру представить уже сгенерированного кодом случайного героя."""
    return (
        "[СИСТЕМА] Игрок выбрал случайного героя: система уже сгенерировала его строго по "
        "правилам PHB 2024 (характеристики, HP, стартовое снаряжение и золото) и записала в лист "
        "персонажа.\n"
        f"{hero_summary(character)}\n"
        "Задание: представь игроку этого героя (1–2 абзаца) — имя, вид, класс, внешность и "
        "характер. Приключение НЕ начинай и пролог не описывай: попроси игрока подтвердить героя "
        "(«да» или кнопка «✅ Подтвердить героя»)."
    )


def is_random_hero_request(text: str) -> bool:
    """Просит ли игрок сгенерировать героя вместо того, чтобы описывать своего."""
    lowered = text.strip().lower()
    if lowered.startswith(("/hero", "/randomhero")):
        return True
    return any(marker in lowered for marker in RANDOM_HERO_MARKERS)


def is_hero_confirmation(text: str) -> bool:
    """Короткое согласие игрока с созданным героем («да», «подтверждаю», «начинаем»)."""
    if len(text) > 40:
        return False
    return bool(HERO_CONFIRM_PATTERN.match(text.strip()))


def detect_hero_details(text: str) -> dict[str, str]:
    """Достаёт из сообщения игрока имя, вид и класс героя (страховка для служебного блока).

    Возвращает словарь с ключами "name"/"race"/"class_name" — только то, что нашлось.
    """
    details: dict[str, str] = {}

    name_match = HERO_NAME_PATTERN.search(text)
    if name_match is not None:
        raw_name = name_match.group("name").strip()
        details["name"] = raw_name[:1].upper() + raw_name[1:]

    species = species_name(text)
    if species is not None:
        details["race"] = species

    class_name = class_name_from_text(text)
    if class_name is not None:
        details["class_name"] = class_name

    return details


def auto_fill_hero_details(character: Character, text: str) -> list[str]:
    """Страховка на случай, если Мастер забудет служебный блок.

    Дополняются ТОЛЬКО пустые поля листа, причём вид и класс — лишь когда игрок назвал
    оба (случайное упоминание «мага» в рассказе не должно сделать героя волшебником).
    """
    if character.is_created:
        return []

    details = detect_hero_details(text)
    notes: list[str] = []

    if "name" in details and character.name == DEFAULT_NAME:
        character.name = details["name"][:64]
        notes.append(f"📛 Имя персонажа: {character.name}.")

    if "race" in details and "class_name" in details:
        if character.race == DEFAULT_RACE:
            character.race = details["race"][:64]
            notes.append(f"🧬 Раса: {character.race}.")
        if character.class_name == DEFAULT_CLASS:
            character.class_name = details["class_name"][:64]
            notes.append(f"⚔️ Класс: {character.class_name} (кость хитов d{character.hit_die}).")

    return notes


def _roll_ability_score() -> int:
    """Характеристика методом «4d6 без наименьшего кубика» (официальный метод PHB 2024)."""
    dice = sorted((_rng.randint(1, 6) for _ in range(4)), reverse=True)
    return sum(dice[:3])


def build_random_hero() -> Character:
    """Собирает случайного героя 1-го уровня строго по правилам PHB 2024 — без участия модели.

    Имя и описание берутся из заготовок, вид и класс — из официальных списков, характеристики
    бросаются методом 4d6 без наименьшего кубика и раскладываются по приоритету класса,
    снаряжение и золото — стартовый набор класса 1-го уровня.
    """
    name, description = _rng.choice(RANDOM_HERO_PROFILES)
    species = _rng.choice(tuple(SPECIES.values()))["name"]
    class_key = _rng.choice(tuple(CLASSES))
    class_name = CLASSES[class_key]["name"]

    scores = sorted((_roll_ability_score() for _ in ABILITIES), reverse=True)
    abilities = dict(zip(ability_priority_for_class(class_name), scores))

    return Character(
        name=name,
        race=species,
        class_name=class_name,
        level=1,
        abilities=abilities,
        inventory=list(starter_equipment_for_class(class_name)),
        gp=starting_gold_for_class(class_name),
        description=description,
        # max_hp/current_hp не задаём: __post_init__ посчитает максимум кости хитов + ТЕЛ.
    )


def apply_starter_loadout(character: Character, recalc_hp: bool = False) -> list[str]:
    """Доводит только что созданного героя до правил PHB 2024 и возвращает заметки для игрока.

    Применяется исключительно на этапе создания персонажа: если Мастер не передал
    характеристики, код выставляет стандартный набор класса, а пустое снаряжение
    заполняет стартовым набором 1-го уровня вместе со стартовым золотом.

    :param recalc_hp: пересчитать максимум HP (кость хитов + модификатор ТЕЛ), когда Мастер
        сам не задавал HP в служебном блоке.
    """
    notes: list[str] = []

    if all(character.abilities[code] == DEFAULT_ABILITIES[code] for code in ABILITIES):
        character.abilities = standard_array_for_class(character.class_name)
        spread = ", ".join(
            f"{ABILITIES[code]} {character.abilities[code]}" for code in ABILITIES
        )
        notes.append(f"🧠 Характеристики по стандартному набору PHB 2024: {spread}.")

    if not character.inventory:
        character.inventory = list(starter_equipment_for_class(character.class_name))
        notes.append(
            f"🎒 Стартовое снаряжение 1-го уровня: {', '.join(character.inventory)}."
        )
        if character.gp == 0:
            character.gp = starting_gold_for_class(character.class_name)
            notes.append(f"💰 Стартовое золото: {character.gp} gp.")

    if recalc_hp:
        # Максимум HP 1-го уровня = максимум кости хитов + модификатор ТЕЛ.
        new_max_hp = max(1, character.hit_die + character.ability_mod("con"))
        if new_max_hp != character.max_hp:
            character.max_hp = new_max_hp
            character.current_hp = new_max_hp
            notes.append(
                f"❤️ Здоровье 1-го уровня: d{character.hit_die} + модификатор ТЕЛ = "
                f"{character.max_hp} HP."
            )

    return notes


# Как оружие наносит урон: дальнобойное (ЛОВ), фехтовальное (лучшая из СИЛ/ЛОВ)
# или обычное рукопашное (СИЛ).
DAMAGE_ABILITY_RANGED = "ranged"
DAMAGE_ABILITY_FINESSE = "finesse"
DAMAGE_ABILITY_MELEE = "melee"

# Урон импровизированной атаки, если оружия в снаряжении нет.
IMPROVISED_DAMAGE_DICE = "1d4"
IMPROVISED_DAMAGE_TYPE = "дробящий"


def find_inventory_weapon(character: Character) -> Optional[tuple[str, str, str, str]]:
    """
    Ищет в снаряжении игрока оружие из официального справочника PHB 2024.

    :return: (название, кость урона, тип урона, способ нанесения) или None.
        Способ нанесения — одна из констант DAMAGE_ABILITY_*.
    """
    for item in character.inventory:
        lowered = item.strip().lower()
        if not lowered:
            continue
        for name, damage, damage_type, properties, _mastery in WEAPONS:
            if name.lower() not in lowered:
                continue
            props = properties.lower()
            if "боеприпас" in props:
                kind = DAMAGE_ABILITY_RANGED
            elif "фехтовальное" in props:
                kind = DAMAGE_ABILITY_FINESSE
            else:
                kind = DAMAGE_ABILITY_MELEE
            return name, damage, damage_type, kind
    return None


def roll_weapon_damage(character: Character) -> tuple[DiceRoll, str]:
    """
    Бросает урон оружием из снаряжения игрока (с модификатором характеристики).

    Если оружия нет — считает импровизированную атаку (1d4 + СИЛ).

    :return: (бросок, подпись оружия для игрока и Мастера).
    """
    weapon = find_inventory_weapon(character)

    if weapon is None:
        damage_dice = IMPROVISED_DAMAGE_DICE
        damage_type = IMPROVISED_DAMAGE_TYPE
        ability_code = "str"
        label = f"импровизированная атака без оружия ({damage_dice})"
    else:
        name, damage_dice, damage_type, kind = weapon
        if kind == DAMAGE_ABILITY_RANGED:
            ability_code = "dex"
        elif kind == DAMAGE_ABILITY_FINESSE:
            # Фехтовальное оружие: берём лучший из модификаторов СИЛ/ЛОВ.
            ability_code = (
                "dex" if character.ability_mod("dex") > character.ability_mod("str") else "str"
            )
        else:
            ability_code = "str"
        label = f"{name} ({damage_dice}, {damage_type} урон)"

    modifier = character.ability_mod(ability_code)

    dice_match = DICE_PATTERN.match(damage_dice)
    if dice_match is None:
        # Фиксированный урон (например, «1» у духовой трубки) — кубики не бросаем.
        roll = DiceRoll.flat(_as_int(damage_dice) + modifier)
    else:
        roll = make_roll(
            count=int(dice_match.group("count") or 1),
            sides=int(dice_match.group("sides")),
            modifier=modifier,
        )

    return roll, f"{label}, модификатор {ABILITIES[ability_code]} {format_modifier(modifier)}"


# Ключи, по которым распознаётся служебный JSON-блок Мастера.
CONTROL_BLOCK_KEYS = frozenset(
    {
        "xp_gained", "hp_change", "add_items", "remove_items", "gp_change",
        "name", "race", "class_name", "level", "abilities", "description",
        "max_hp", "current_hp", "heroic_inspiration",
    }
)

# Служебный блок в тройных обратных кавычках (``` или ```json).
FENCED_JSON_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _iter_balanced_objects(text: str):
    """Итератор по сбалансированным подстрокам {...} (с учётом строк JSON)."""
    depth = 0
    start: Optional[int] = None
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield start, index + 1, text[start:index + 1]
                start = None


def _try_parse_json(raw: str) -> Optional[dict]:
    """Пытается разобрать JSON-объект; при неудаче возвращает None."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Удаляет из текста указанные диапазоны (с объединением пересечений)."""
    if not spans:
        return text

    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def extract_control_block(text: str) -> tuple[str, Optional[dict]]:
    """
    Вырезает служебный JSON-блок из ответа Мастера.

    Возвращает пару (текст без блока, данные блока или None).
    """
    control: Optional[dict] = None
    spans: list[tuple[int, int]] = []

    # 1) Блоки внутри тройных обратных кавычек (вырезаем всегда, применяем — если валиден).
    for match in FENCED_JSON_PATTERN.finditer(text):
        data = _try_parse_json(match.group(1))
        if data is not None:
            control = data
        spans.append(match.span())

    # 2) «Голые» JSON-объекты (если модель забыла про обратные кавычки).
    for start, end, raw in _iter_balanced_objects(text):
        if any(begin <= start and end <= finish for begin, finish in spans):
            continue
        data = _try_parse_json(raw)
        if data is not None and CONTROL_BLOCK_KEYS.intersection(data.keys()):
            control = data
            spans.append((start, end))

    clean = _strip_spans(text, spans)
    # Убираем «осиротевшие» пустые блоки кода и лишние пустые строки.
    clean = re.sub(r"```(?:json)?\s*```", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return clean.strip(), control


# ---------------------------------------------------------------------------
# Инлайн-клавиатуры: быстрые броски и управление персонажем
# ---------------------------------------------------------------------------

# Сетка кнопок под каждым ответом Мастера (callback_data разбирают хэндлеры ниже).
ACTION_KEYBOARD = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="🎲 d20", callback_data="roll:d20"),
            InlineKeyboardButton(text="🎲 d20 с преим.", callback_data="roll:adv"),
            InlineKeyboardButton(text="🎲 d20 с помех.", callback_data="roll:dis"),
        ],
        [
            InlineKeyboardButton(text="📜 Лист", callback_data="sheet"),
            InlineKeyboardButton(text="🎒 Инвентарь", callback_data="inventory"),
            InlineKeyboardButton(text="🎲 Бросок урона", callback_data="roll:damage"),
        ],
        [
            InlineKeyboardButton(text="🧠 Проверки по статам", callback_data="checks"),
        ],
    ]
)


def build_checks_keyboard(character: Character) -> InlineKeyboardMarkup:
    """Сетка проверок характеристик с модификаторами из листа персонажа игрока."""
    buttons = [
        InlineKeyboardButton(
            text=f"{ABILITIES[code]} {format_modifier(character.ability_mod(code))}",
            callback_data=f"check:{code}",
        )
        for code in ABILITIES
    ]
    rows = [buttons[index:index + 3] for index in range(0, len(buttons), 3)]
    rows.append([InlineKeyboardButton(text="⬅️ Быстрые действия", callback_data="menu")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


# Кнопки этапа создания персонажа: пока герой не подтверждён, показываем именно их.
CREATION_KEYBOARD = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="🎲 Случайный герой", callback_data="hero:random"),
            InlineKeyboardButton(text="✅ Подтвердить героя", callback_data="hero:confirm"),
        ],
        [
            InlineKeyboardButton(text="📜 Лист", callback_data="sheet"),
        ],
    ]
)


def phase_keyboard(character: Character) -> InlineKeyboardMarkup:
    """Клавиатура по фазе игры: создание героя или обычные игровые кнопки."""
    return ACTION_KEYBOARD if character.hero_confirmed else CREATION_KEYBOARD


# ---------------------------------------------------------------------------
# 7. БАЗА ДАННЫХ SQLITE (постоянное хранение листов и истории)
# ---------------------------------------------------------------------------

# Схема создаётся при первом подключении; IF NOT EXISTS — безопасно повторять.
DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS characters (
    user_id    INTEGER PRIMARY KEY,
    data       TEXT      NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history (user_id, id);
"""


class BotDatabase:
    """
    Постоянное хранилище состояния бота в локальном файле SQLite.

    Таблицы:
        * characters   — сериализованный Character (JSON), по одной записи на игрока;
        * chat_history — все сообщения игрока и Мастера (роли user / assistant).

    Соединение открывается лениво при первом обращении, переиспользуется и
    защищено блокировкой, поэтому методы безопасно вызывать и из обработчиков
    aiogram, и из отдельных потоков. Локальные операции SQLite занимают доли
    миллисекунды, так что цикл событий они практически не блокируют.
    """

    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    # --- подключение ---

    def _connect(self) -> sqlite3.Connection:
        """Открывает соединение (один раз) и применяет схему. Вызывать под self._lock."""
        if self._conn is not None:
            return self._conn

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: доступ сериализуем собственным Lock'ом,
        # поэтому соединение можно при необходимости трогать из другого потока.
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL + NORMAL — быстрая и безопасная запись для локального файла.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(DB_SCHEMA)
        conn.commit()
        self._conn = conn
        return conn

    def init(self) -> None:
        """Создаёт файл базы, таблицы и индексы (идемпотентно)."""
        with self._lock:
            self._connect()
        logger.info("База данных готова: %s", self.path)

    def close(self) -> None:
        """Закрывает соединение с базой (вызывается при остановке бота)."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # --- таблица characters ---

    def save_character(self, user_id: int, character: Character) -> None:
        """Сохраняет (или обновляет) лист персонажа игрока."""
        payload = character.to_json()
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT INTO characters (user_id, data, updated_at) "
                "VALUES (?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(user_id) DO UPDATE SET "
                "data = excluded.data, updated_at = CURRENT_TIMESTAMP",
                (int(user_id), payload),
            )
            conn.commit()

    def load_character(self, user_id: int) -> Optional[Character]:
        """Возвращает сохранённый лист персонажа или None, если записи ещё нет."""
        with self._lock:
            row = self._connect().execute(
                "SELECT data FROM characters WHERE user_id = ?",
                (int(user_id),),
            ).fetchone()
        if row is None:
            return None
        return Character.from_json(row["data"])

    # --- таблица chat_history ---

    def append_message(self, user_id: int, role: str, content: str) -> None:
        """Добавляет одно сообщение (user/assistant) в историю игрока."""
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT INTO chat_history (user_id, role, content) VALUES (?, ?, ?)",
                (int(user_id), str(role), str(content)),
            )
            conn.commit()

    def load_history(
        self,
        user_id: int,
        limit: int = MAX_HISTORY_MESSAGES,
    ) -> list[dict[str, str]]:
        """Возвращает последние `limit` сообщений игрока в хронологическом порядке."""
        with self._lock:
            rows = self._connect().execute(
                "SELECT role, content FROM chat_history WHERE user_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (int(user_id), max(0, int(limit))),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    def clear_user(self, user_id: int) -> None:
        """Удаляет историю и лист персонажа игрока (команды /start и /reset)."""
        with self._lock:
            conn = self._connect()
            with conn:  # атомарно: либо удалилось всё, либо ничего
                conn.execute("DELETE FROM chat_history WHERE user_id = ?", (int(user_id),))
                conn.execute("DELETE FROM characters WHERE user_id = ?", (int(user_id),))


# Единственный на процесс экземпляр базы (файл bot_database.db в корне проекта).
db = BotDatabase()


# ---------------------------------------------------------------------------
# 8. ПАМЯТЬ ДИАЛОГА И СЕССИИ (на каждого пользователя отдельно)
# ---------------------------------------------------------------------------


class Session:
    """
    Сессия одного пользователя: история диалога и лист персонажа.

    В памяти лежат только последние MAX_HISTORY_MESSAGES сообщений,
    чтобы не переполнять контекст модели и не жечь токены. Каждое изменение
    сразу дублируется в SQLite, поэтому при перезапуске бота прогресс игрока
    поднимается из базы (см. Session.restore).
    """

    __slots__ = ("user_id", "history", "character")

    def __init__(self, user_id: int) -> None:
        self.user_id: int = int(user_id)
        self.history: deque[dict[str, str]] = deque(maxlen=MAX_HISTORY_MESSAGES)
        self.character: Character = Character()

    @classmethod
    def restore(cls, user_id: int) -> "Session":
        """Поднимает сессию из базы: лист персонажа и последние сообщения."""
        session = cls(user_id)
        stored = db.load_character(user_id)
        if stored is None:
            # Первое обращение игрока — фиксируем в базе героя по умолчанию.
            db.save_character(user_id, session.character)
        else:
            session.character = stored
        for message in db.load_history(user_id):
            session.history.append(message)
        return session

    def add(self, role: str, content: str) -> None:
        """Добавляет сообщение роли 'user' или 'assistant' в историю (память + база)."""
        self.history.append({"role": role, "content": content})
        db.append_message(self.user_id, role, content)

    def clear(self) -> None:
        """Полностью очищает историю и создаёт нового персонажа (память + база)."""
        self.history.clear()
        self.character = Character()
        db.clear_user(self.user_id)
        db.save_character(self.user_id, self.character)

    def save_character(self) -> None:
        """Сохраняет текущий лист персонажа в базу (после apply_control и т.п.)."""
        db.save_character(self.user_id, self.character)

    def messages(self) -> list[dict[str, str]]:
        """Снимок истории для передачи в API (без системного промпта)."""
        return list(self.history)


# user_id -> Session
_sessions: dict[int, Session] = {}


def get_session(user_id: int) -> Session:
    """Возвращает сессию пользователя, при первом обращении поднимая её из базы."""
    session = _sessions.get(user_id)
    if session is None:
        session = Session.restore(user_id)
        _sessions[user_id] = session
        logger.info(
            "Сессия пользователя %s загружена из базы (сообщений: %d, уровень героя: %d)",
            user_id,
            len(session.history),
            session.character.level,
        )
    return session


# ---------------------------------------------------------------------------
# 9. КЛИЕНТ DEEPSEEK (через официальный SDK openai)
# ---------------------------------------------------------------------------

# Клиент создаётся один раз; реальный ключ проверяется при запуске в main().
openai_client: Optional[AsyncOpenAI] = (
    AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    if DEEPSEEK_API_KEY
    else None
)


async def ask_dungeon_master(history: Iterable[dict[str, str]]) -> str:
    """
    Отправляет историю диалога Мастеру и возвращает текст ответа.

    Системный промпт добавляется к каждому запросу, а сама история уже
    ограничена по длине (см. Session).

    :raise RuntimeError: если клиент не инициализирован или ответ пуст.
    """
    if openai_client is None:
        raise RuntimeError("DEEPSEEK_API_KEY не задан — клиент DeepSeek недоступен.")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
    ]

    response = await openai_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=DM_TEMPERATURE,
        max_tokens=DM_MAX_TOKENS,
        stream=False,
    )

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("Мастер вернул пустой ответ.")

    return content.strip()


# ---------------------------------------------------------------------------
# 10. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ВЫВОДА
# ---------------------------------------------------------------------------


def split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Разбивает длинный текст на части по лимиту Telegram (с переносом по абзацам)."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip("\n ")
    return chunks


async def send_long_message(
    message: Message,
    text: str,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
) -> None:
    """Отправляет (возможно, длинный) текст, разбивая его на несколько сообщений.

    Инлайн-клавиатура (если задана) прикрепляется к последнему сообщению.
    """
    chunks = [chunk for chunk in split_message(text) if chunk]
    for index, chunk in enumerate(chunks):
        markup = reply_markup if index == len(chunks) - 1 else None
        await message.answer(chunk, reply_markup=markup)


# ---------------------------------------------------------------------------
# 11. ОБРАБОТЧИКИ (aiogram router)
# ---------------------------------------------------------------------------

router = Router()


async def _answer_with_dungeon_master(
    message: Message,
    session: Session,
    hero_card_title: Optional[str] = None,
) -> None:
    """Запрашивает ответ Мастера, обновляет лист персонажа и отправляет ответ игроку.

    Все изменения (сообщения и лист персонажа) сразу попадают в SQLite.

    :param hero_card_title: заголовок карточки героя, пока он не подтверждён (например,
        «🎲 Случайный герой готов!»). По умолчанию подбирается по ситуации.
    """
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    try:
        raw_reply = await ask_dungeon_master(session.messages())
    except APIError as error:
        logger.error("Ошибка DeepSeek API: %s", error)
        await message.answer(API_ERROR_TEXT)
        return
    except Exception:  # noqa: BLE001 — на верхнем уровне бота логируем всё непредвиденное
        logger.exception("Непредвиденная ошибка при обращении к Мастеру")
        await message.answer(GENERIC_ERROR_TEXT)
        return

    # 1) Отделяем служебный блок изменений от текста и применяем его к листу персонажа.
    reply, control = extract_control_block(raw_reply)
    notes = session.character.apply_control(control) if control else []

    starter_notes: list[str] = []
    if session.character.is_created and not session.character.hero_confirmed:
        # Герой описан, но ещё не подтверждён: доводим лист до правил PHB 2024
        # (характеристики, максимум HP и стартовый набор снаряжения 1-го уровня).
        # Идемпотентно: уже заполненные значения код не трогает.
        hp_explicit = bool(
            control
            and (control.get("max_hp") is not None or control.get("current_hp") is not None)
        )
        starter_notes = apply_starter_loadout(session.character, recalc_hp=not hp_explicit)
        notes.extend(starter_notes)

    if control is not None:
        logger.info("Служебный блок Мастера применён: %s", control)
    if control is not None or starter_notes:
        # Сразу фиксируем изменения листа в SQLite, чтобы прогресс не потерялся.
        session.save_character()

    if not reply:
        # Модель вернула только служебный блок — не оставляем игрока без реплики.
        reply = "Мастер молчаливо следит за происходящим.\n\nЧто ты делаешь?"

    # 2) В память диалога кладём ТОЛЬКО чистый текст (без служебного JSON).
    session.add("assistant", reply)

    # 3) Отправляем ответ Мастера с кнопками текущей фазы (создание героя или игра).
    await send_long_message(message, reply, reply_markup=phase_keyboard(session.character))

    # 4) …и сообщаем игроку об изменениях листа персонажа.
    if notes:
        body = "\n".join(f"• {note}" for note in notes)
        await message.answer(f"📈 Обновление листа персонажа:\n{body}")

    # 5) Пока герой не подтверждён, показываем лист и ждём подтверждения —
    #    приключение после этого шага начинают только по воле игрока.
    if session.character.is_created and not session.character.hero_confirmed:
        if hero_card_title is not None:
            await _send_hero_card(message, session, hero_card_title)
        elif starter_notes:
            await _send_hero_card(message, session, HERO_CARD_TITLE_CREATED)
        else:
            await _send_hero_card(message, session)


async def _resolve_roll(
    message: Message,
    session: Session,
    roll: DiceRoll,
    context: Optional[str] = None,
) -> None:
    """Обрабатывает бросок, сделанный кодом бота (команда /roll или кнопка).

    1) показывает игроку «математику» броска;
    2) пишет бросок в историю как ход игрока — в память и в SQLite;
    3) просит Мастера описать исход.

    :param context: служебное сообщение для Мастера; по умолчанию — стандартное описание броска.
    """
    await message.answer(f"🎲 {roll.describe()}")
    logger.info("Пользователь %s бросил %s = %s", session.user_id, roll.notation, roll.total)

    session.add("user", context or roll.context_message())
    await _answer_with_dungeon_master(message, session)


def _callback_context(callback: CallbackQuery) -> Optional[tuple[Message, int]]:
    """Достаёт сообщение и id игрока из нажатия кнопки: (Message, user_id) или None.

    None означает, что апдейт недоступен (например, сообщение слишком старое).
    """
    if callback.from_user is None or not isinstance(callback.message, Message):
        return None
    return callback.message, callback.from_user.id


async def _begin_hero_creation(message: Message, session: Session) -> None:
    """Этап 1: просит Мастера поприветствовать игрока и создать героя (приключение не начинается)."""
    session.add("user", HERO_CREATION_START_PROMPT)
    await _answer_with_dungeon_master(message, session)


async def _send_hero_card(
    message: Message,
    session: Session,
    title: str = HERO_CARD_TITLE_PENDING,
) -> None:
    """Показывает лист созданного героя и просит игрока подтвердить его."""
    await send_long_message(
        message,
        f"{title}\n\n{format_character_sheet(session.character)}\n\n{HERO_CARD_QUESTION}",
        reply_markup=CREATION_KEYBOARD,
    )


async def _create_random_hero(message: Message, session: Session) -> None:
    """Генерирует случайного героя кодом по правилам PHB 2024 и просит Мастера его представить."""
    hero = build_random_hero()

    # Имя и описание, которые игрок успел назвать сам, не теряем.
    if session.character.name != DEFAULT_NAME:
        hero.name = session.character.name
    if session.character.description:
        hero.description = session.character.description

    session.character = hero
    session.save_character()
    logger.info(
        "Пользователь %s получил случайного героя: %s, %s %s",
        session.user_id,
        hero.name,
        hero.race,
        hero.class_name,
    )

    session.add("user", random_hero_prompt(hero))
    await _answer_with_dungeon_master(message, session, hero_card_title=HERO_CARD_TITLE_RANDOM)


async def _confirm_hero_and_start_prologue(message: Message, session: Session) -> None:
    """Игрок подтвердил героя: фиксируем это и начинаем вводную сцену пролога."""
    session.character.hero_confirmed = True
    session.save_character()

    session.add("user", prologue_prompt(session.character))
    logger.info("Пользователь %s подтвердил героя — начинается пролог", session.user_id)
    await _answer_with_dungeon_master(message, session)


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    """/start — приветствие, сброс прошлой сессии и создание нового героя."""
    if message.from_user is None:
        return
    user = message.from_user
    session = get_session(user.id)
    session.clear()

    await message.answer(WELCOME_TEXT)
    await _begin_hero_creation(message, session)
    logger.info("Пользователь %s начал создание персонажа", user.id)


@router.message(Command("reset"))
async def handle_reset(message: Message) -> None:
    """/reset — принудительный сброс контекста и создание героя с чистого листа."""
    if message.from_user is None:
        return
    user = message.from_user
    session = get_session(user.id)
    session.clear()

    await message.answer(
        "🔄 Контекст полностью сброшен: прошлый герой и история удалены.\n"
        "Создаём нового героя…"
    )
    await _begin_hero_creation(message, session)
    logger.info("Пользователь %s сбросил сессию", user.id)


@router.message(Command("hero"))
async def handle_random_hero(message: Message) -> None:
    """/hero — сгенерировать случайного героя 1-го уровня по правилам PHB 2024."""
    if message.from_user is None:
        return
    session = get_session(message.from_user.id)
    if session.character.hero_confirmed:
        await message.answer(HERO_ALREADY_CONFIRMED_TEXT, reply_markup=ACTION_KEYBOARD)
        return

    await _create_random_hero(message, session)
    logger.info("Пользователь %s вызвал случайного героя командой", message.from_user.id)


@router.message(Command("sheet"))
async def handle_sheet(message: Message) -> None:
    """/sheet — красиво форматирует и отправляет текущий лист персонажа."""
    if message.from_user is None:
        return
    session = get_session(message.from_user.id)
    await send_long_message(
        message,
        format_character_sheet(session.character),
        reply_markup=phase_keyboard(session.character),
    )
    logger.info("Пользователь %s открыл лист персонажа", message.from_user.id)


@router.message(Command("inventory"))
async def handle_inventory(message: Message) -> None:
    """/inventory — компактный список снаряжения и золота."""
    if message.from_user is None:
        return
    session = get_session(message.from_user.id)
    await message.answer(
        format_inventory(session.character),
        reply_markup=phase_keyboard(session.character),
    )
    logger.info("Пользователь %s открыл снаряжение", message.from_user.id)


@router.message(Command("check"))
async def handle_check(message: Message) -> None:
    """/check — меню проверок характеристик (СИЛ/ЛОВ/ТЕЛ/ИНТ/МУД/ХАР)."""
    if message.from_user is None:
        return
    session = get_session(message.from_user.id)
    await message.answer(
        CHECKS_MENU_TEXT,
        reply_markup=build_checks_keyboard(session.character),
    )
    logger.info("Пользователь %s открыл меню проверок", message.from_user.id)


@router.callback_query(F.data == "hero:random")
async def handle_random_hero_button(callback: CallbackQuery) -> None:
    """Инлайн-кнопка «🎲 Случайный герой»: код собирает героя по правилам PHB 2024."""
    target = _callback_context(callback)
    if target is None:
        await callback.answer(STALE_CALLBACK_TEXT, show_alert=True)
        return
    # Гасим «часики» на кнопке — обязательно для любой CallbackQuery.
    await callback.answer()

    message, user_id = target
    session = get_session(user_id)
    if session.character.hero_confirmed:
        await callback.answer(HERO_ALREADY_CONFIRMED_ALERT, show_alert=True)
        return

    logger.info("Пользователь %s нажал кнопку «Случайный герой»", user_id)
    await _create_random_hero(message, session)


@router.callback_query(F.data == "hero:confirm")
async def handle_hero_confirm_button(callback: CallbackQuery) -> None:
    """Инлайн-кнопка «✅ Подтвердить героя»: герой принят, начинается пролог."""
    target = _callback_context(callback)
    if target is None:
        await callback.answer(STALE_CALLBACK_TEXT, show_alert=True)
        return

    message, user_id = target
    session = get_session(user_id)

    if session.character.hero_confirmed:
        await callback.answer(HERO_ALREADY_CONFIRMED_ALERT, show_alert=True)
        return
    if not session.character.is_created:
        await callback.answer(HERO_NOT_CREATED_ALERT, show_alert=True)
        return
    await callback.answer()

    logger.info("Пользователь %s подтвердил героя кнопкой", user_id)
    await _confirm_hero_and_start_prologue(message, session)


@router.callback_query(F.data == "sheet")
async def handle_sheet_button(callback: CallbackQuery) -> None:
    """Инлайн-кнопка «📜 Лист» под игровыми ответами Мастера."""
    target = _callback_context(callback)
    if target is None:
        await callback.answer(STALE_CALLBACK_TEXT, show_alert=True)
        return
    # Гасим «часики» на кнопке — обязательно для любой CallbackQuery.
    await callback.answer()

    message, user_id = target
    session = get_session(user_id)
    await send_long_message(
        message,
        format_character_sheet(session.character),
        reply_markup=phase_keyboard(session.character),
    )
    logger.info("Пользователь %s открыл лист персонажа кнопкой", user_id)


@router.callback_query(F.data == "inventory")
async def handle_inventory_button(callback: CallbackQuery) -> None:
    """Инлайн-кнопка «🎒 Инвентарь»: снаряжение и золото персонажа."""
    target = _callback_context(callback)
    if target is None:
        await callback.answer(STALE_CALLBACK_TEXT, show_alert=True)
        return
    await callback.answer()

    message, user_id = target
    session = get_session(user_id)
    await message.answer(
        format_inventory(session.character),
        reply_markup=phase_keyboard(session.character),
    )
    logger.info("Пользователь %s открыл снаряжение кнопкой", user_id)


@router.callback_query(F.data == "checks")
async def handle_checks_button(callback: CallbackQuery) -> None:
    """Инлайн-кнопка «🧠 Проверки по статам»: меню проверок характеристик."""
    target = _callback_context(callback)
    if target is None:
        await callback.answer(STALE_CALLBACK_TEXT, show_alert=True)
        return
    await callback.answer()

    message, user_id = target
    session = get_session(user_id)
    await message.answer(
        CHECKS_MENU_TEXT,
        reply_markup=build_checks_keyboard(session.character),
    )
    logger.info("Пользователь %s открыл меню проверок кнопкой", user_id)


@router.callback_query(F.data == "menu")
async def handle_menu_button(callback: CallbackQuery) -> None:
    """Инлайн-кнопка «⬅️ Быстрые действия»: возвращает основную сетку кнопок."""
    target = _callback_context(callback)
    if target is None:
        await callback.answer(STALE_CALLBACK_TEXT, show_alert=True)
        return
    await callback.answer()

    message, _user_id = target
    await message.answer(ACTION_MENU_TEXT, reply_markup=ACTION_KEYBOARD)


# Кнопки бросков: callback_data -> (режим d20, подпись для Мастера).
D20_BUTTON_MODES: dict[str, str] = {"d20": "", "adv": "advantage", "dis": "disadvantage"}
D20_BUTTON_PURPOSES: dict[str, str] = {
    "d20": "бросок d20",
    "adv": "бросок d20 с преимуществом",
    "dis": "бросок d20 с помехой",
}


@router.callback_query(F.data.startswith("roll:"))
async def handle_roll_button(callback: CallbackQuery) -> None:
    """Кнопки быстрых бросков: d20, d20 с преимуществом/помехой и бросок урона.

    Бросок считает код, результат показывается игроку, сохраняется в историю
    (память + SQLite) и передаётся Мастеру для описания исхода.
    """
    action = (callback.data or "").split(":", 1)[-1]

    target = _callback_context(callback)
    if target is None:
        await callback.answer(STALE_CALLBACK_TEXT, show_alert=True)
        return
    if action not in D20_BUTTON_PURPOSES and action != "damage":
        await callback.answer(UNKNOWN_BUTTON_TEXT, show_alert=True)
        return
    # Гасим «часики» на кнопке — обязательно для любой CallbackQuery.
    await callback.answer()

    message, user_id = target
    session = get_session(user_id)

    if action == "damage":
        roll, label = roll_weapon_damage(session.character)
        context = roll.context_message(f"бросок урона ({label})")
    else:
        mode = D20_BUTTON_MODES[action]
        roll = make_roll(count=2 if mode else 1, sides=20, mode=mode)
        context = roll.context_message(D20_BUTTON_PURPOSES[action])

    logger.info("Пользователь %s нажал кнопку броска «%s»", user_id, action)
    await _resolve_roll(message, session, roll, context=context)


@router.callback_query(F.data.startswith("check:"))
async def handle_check_button(callback: CallbackQuery) -> None:
    """Инлайн-кнопка характеристики: d20 + модификатор из листа персонажа игрока."""
    code = (callback.data or "").split(":", 1)[-1]

    target = _callback_context(callback)
    if target is None:
        await callback.answer(STALE_CALLBACK_TEXT, show_alert=True)
        return
    if code not in ABILITY_FULL_RU:
        await callback.answer(UNKNOWN_BUTTON_TEXT, show_alert=True)
        return
    await callback.answer()

    message, user_id = target
    session = get_session(user_id)
    character = session.character

    ability_name = ABILITY_FULL_RU[code]
    ability_genitive = ABILITY_GENITIVE_RU[code]
    modifier = character.ability_mod(code)
    roll = make_roll(count=1, sides=20, modifier=modifier)

    # Служебное сообщение Мастеру ровно в оговорённом виде.
    context = (
        f"Игрок совершил проверку {ability_genitive}: 1d20 ({roll.rolls[0]}) "
        f"{format_modifier(modifier)} = {roll.total}. {DM_ROLL_INSTRUCTION}"
    )

    logger.info(
        "Пользователь %s прошёл проверку %s: d20(%s)%s = %s",
        user_id,
        ability_name,
        roll.rolls[0],
        format_modifier(modifier),
        roll.total,
    )
    await _resolve_roll(message, session, roll, context=context)


@router.message(Command("roll"))
async def handle_roll(message: Message, command: CommandObject) -> None:
    """/roll <кубик> — бросок кубиков кодом с описанием исхода Мастером."""
    if message.from_user is None:
        return
    user = message.from_user
    expression = (command.args or "").strip()

    if not expression:
        await message.answer(ROLL_USAGE_TEXT)
        return

    try:
        roll = parse_and_roll(expression)
    except ValueError as error:
        await message.answer(f"⚠️ {error}\n\n{ROLL_USAGE_TEXT}")
        return

    await _resolve_roll(message, get_session(user.id), roll)


@router.message(F.text & ~F.text.startswith("/"))
async def handle_player_action(message: Message) -> None:
    """Любое текстовое сообщение: создание героя, его подтверждение или действие персонажа."""
    text = (message.text or "").strip()
    if not text or message.from_user is None:
        return

    if len(text) > MAX_PLAYER_INPUT:
        text = text[:MAX_PLAYER_INPUT]

    user = message.from_user
    session = get_session(user.id)
    session.add("user", text)

    stage = creation_stage(session.character)

    # ЭТАП 1: пока герой не описан, приключение не начинается — только создание персонажа.
    if stage == CREATION_STAGE_HERO:
        # Страховка: если Мастер забудет служебный блок, имя, вид и класс распознает код.
        fallback_notes = auto_fill_hero_details(session.character, text)
        if fallback_notes:
            session.save_character()
            logger.info("Лист персонажа дополнен кодом: %s", fallback_notes)
            await message.answer(
                "📈 Обновление листа персонажа:\n"
                + "\n".join(f"• {note}" for note in fallback_notes)
            )

        if not session.character.is_created and is_random_hero_request(text):
            await _create_random_hero(message, session)
            return

        if session.character.is_created and is_hero_confirmation(text):
            # Игрок сразу назвал героя и подтвердил его — пролог начинается без паузы.
            await _confirm_hero_and_start_prologue(message, session)
            return

        session.add("user", creation_prompt(session))
        await _answer_with_dungeon_master(message, session)
        return

    # Герой описан: ждём подтверждения игрока и только потом начинаем приключение.
    if stage == CREATION_STAGE_CONFIRM:
        if is_hero_confirmation(text):
            await _confirm_hero_and_start_prologue(message, session)
            return

        session.add("user", hero_confirmation_prompt(session.character))
        await _answer_with_dungeon_master(message, session)
        return

    await _answer_with_dungeon_master(message, session)


@router.message()
async def handle_unsupported(message: Message) -> None:
    """Заглушка для нетекстовых сообщений (фото, стикеры и т.п.)."""
    if message.from_user is None:
        return
    session = get_session(message.from_user.id)
    if not session.character.hero_confirmed:
        await message.answer(
            "Я понимаю только текст и кнопки. Опиши героя словами (имя, вид, класс) "
            "или нажми «🎲 Случайный герой», а затем «✅ Подтвердить героя».",
            reply_markup=CREATION_KEYBOARD,
        )
        return

    await message.answer(
        "Я понимаю только текст и кнопки. Опиши своё действие словами, нажми кнопку "
        "быстрого броска (🎲 d20, 🎲 Бросок урона, 🧠 Проверки по статам) "
        "или используй команды /roll, /sheet, /inventory, /check.",
        reply_markup=ACTION_KEYBOARD,
    )


# ---------------------------------------------------------------------------
# 12. ЗАПУСК БОТА
# ---------------------------------------------------------------------------

BOT_COMMANDS = [
    BotCommand(command="start", description="Начать новую игру и создать героя"),
    BotCommand(command="reset", description="Сбросить контекст и создать героя заново"),
    BotCommand(command="hero", description="Случайный герой 1-го уровня по правилам PHB 2024"),
    BotCommand(command="roll", description="Бросить кубик, например d20 или 2d6+3"),
    BotCommand(command="sheet", description="Показать лист персонажа"),
    BotCommand(command="inventory", description="Показать снаряжение и золото"),
    BotCommand(command="check", description="Проверки характеристик с модификатором"),
]


async def main() -> None:
    """Точка входа: проверяет конфиг, поднимает polling и корректно всё закрывает."""
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit(
            "Не задан TELEGRAM_BOT_TOKEN.\n"
            "Создайте файл .env на основе .env.example и укажите токен от @BotFather."
        )
    if not DEEPSEEK_API_KEY:
        raise SystemExit(
            "Не задан DEEPSEEK_API_KEY.\n"
            "Создайте файл .env на основе .env.example и укажите ключ DeepSeek API."
        )

    # parse_mode=None: ответы Мастера — «сырой» текст, чтобы разметка модели
    # не ломала отправку сообщений. Промпт просит писать без Markdown.
    bot = Bot(token=TELEGRAM_BOT_TOKEN, default=DefaultBotProperties(parse_mode=None))
    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    # Готовим постоянное хранилище: создаём bot_database.db, таблицы и индексы.
    db.init()

    try:
        await bot.set_my_commands(BOT_COMMANDS)
        logger.info("Бот запущен. Нажмите Ctrl+C для остановки.")
        # close_bot_session=True (по умолчанию) закрывает HTTP-сессию бота на выходе.
        await dispatcher.start_polling(
            bot,
            allowed_updates=dispatcher.resolve_used_update_types(),
        )
    finally:
        # Корректно закрываем все сетевые сессии и соединение с базой.
        if openai_client is not None:
            await openai_client.close()
        await bot.session.close()
        db.close()
        logger.info("Соединения закрыты. До встречи в подземелье!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit) as exc:
        # SystemExit при отсутствии ключей / корректная остановка по Ctrl+C.
        if str(exc):
            print(exc)


