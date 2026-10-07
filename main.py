"""
Telegram-бот «Dungeon Master» для настольной ролевой игры Dungeons & Dragons 5e (2024 / «5.5e»).

Стек:
    * Python 3
    * aiogram 3.x      — асинхронный фреймворк для Telegram Bot API
    * openai (SDK)       — обращение к официальному API DeepSeek (OpenAI-совместимый)
    * python-dotenv    — загрузка переменных окружения из файла .env
    * sqlite3 (stdlib) — постоянное хранение сессий в файле bot_database.db
      (рядом с main.py или в каталоге из переменной окружения DATA_DIR — см. ниже)

Команды:
    /start     — приветствие, сброс прошлой сессии и создание героя
    /reset     — принудительный сброс контекста, игра начинается с чистого листа
    /hero      — сгенерировать случайного героя 1-го уровня по правилам PHB 2024
    /roll      — бросок кубиков, считаемый кодом (например: /roll d20, /roll 2d6+3)
    /sheet     — показать лист персонажа (имя, класс, HP, характеристики, снаряжение)
    /inventory — список снаряжения и золота
    /check     — меню проверок характеристик (СИЛ/ЛОВ/ТЕЛ/ИНТ/МУД/ХАР)
    /spells    — книга заклинаний: ячейки, применение и подготовка заклинаний
    /rest      — отдых: короткий (1 час) и продолжительный (8 часов), восстановление ячеек
    /debug_tokens — СКРЫТАЯ команда (её нет в меню Telegram): расход токенов DeepSeek
                    за текущую сессию и размер последнего запроса. Только администраторы
                    из ADMIN_USER_IDS (см. handle_debug_tokens)

ЭТАП 0 — выбор сеттинга (мира игры):
    Прежде чем описывать героя, игрок выбирает мир кнопками «🎲 Забытые Королевства (D&D)»
    или «⚔️ Вселенная Warcraft (Азерот)» (callback_data «setting:dnd_classic» /
    «setting:warcraft»). Выбор хранится в поле setting листа персонажа. Для сеттинга
    Warcraft в системный промпт Мастера подмешивается хроника Азерота из файла
    warcraft_lore.txt (сжатая версия; полный текст — «вов.txt», см. build_system_prompt):
    правила боёвки, броски и лист персонажа остаются по D&D 2024, а мир, фракции, локации,
    монстры и NPC берутся строго из хроники (эпоха Третьей Войны, 20–27 гг. ADP).

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
    🎲 d20 / 🎲 d20 с преим. / 🎲 d20 с помех. — мгновенный бросок атаки/проверки кодом
    🗡 1d6 / 1d8 / 1d10 / 1d12                — бросок урона указанной костью (+ мод. оружия)
    📜 Лист / 🎒 Инвентарь / 🎲 Бросок урона    — лист, снаряжение, урон оружием
    🧠 Проверки по статам                      — меню проверок d20 + модификатор
    🎲 Случайный герой / ✅ Подтвердить героя    — кнопки этапа создания персонажа
    🌍 Выбор мира                              — повторный выбор сеттинга при создании героя
    🎲 Забытые Королевства / ⚔️ Вселенная Warcraft — выбор сеттинга (мира игры) в начале
                                                  создания героя; «setting:dnd_classic»
                                                  и «setting:warcraft» в callback_data
    Любое нажатие пишется в историю и SQLite так же, как обычная команда игрока.

Любое другое текстовое сообщение воспринимается как действие игрока
и передаётся Мастеру вместе с историей диалога.

Хранение данных:
    В локальной базе bot_database.db (в корне проекта) две таблицы:
        * characters   — сериализованный лист персонажа, по одному на игрока; колонки
                         location, quest и setting дублируют сводку HUD (локация, цель)
                         и выбранный сеттинг партии для удобной отладки SQL-запросами
                         (источник истины — JSON в колонке data);
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
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

from dotenv import load_dotenv

from aiogram import BaseMiddleware, Bot, Dispatcher, F, Router
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

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, ValidationError

from dnd2024_reference import (
    ABILITIES,
    ABILITY_FULL_RU,
    ABILITY_GENITIVE_RU,
    ARMOR,
    CLASSES,
    MAX_LEVEL,
    SHIELD_BONUS,
    SKILLS,
    SPELL_LEVEL_RU,
    SPECIES,
    WEAPONS,
    ability_modifier,
    ability_priority_for_class,
    average_hit_points,
    build_reference_digest,
    class_cantrip_list,
    class_name_from_text,
    class_spell_list,
    default_cantrips_for_class,
    default_known_spells_for_class,
    format_modifier,
    hit_die_sides,
    is_pact_caster,
    is_spellcaster_class,
    is_spontaneous_caster,
    max_prepared_spells,
    next_xp_threshold,
    normalize_ability_key,
    proficiency_bonus,
    resolve_class_key,
    species_name,
    species_traits,
    spell_level,
    spell_slots_for_level,
    spellcasting_ability_for_class,
    standard_array_for_class,
    starter_equipment_for_class,
    starting_gold_for_class,
)
from prompts import (
    BASE_DM_PROMPT,
    CONTROL_BLOCK_INSTRUCTIONS,
    DM_ROLL_INSTRUCTION,
    HERO_ALREADY_CONFIRMED_ALERT,
    HERO_ALREADY_CONFIRMED_TEXT,
    HERO_CARD_QUESTION,
    HERO_CARD_TITLE_CREATED,
    HERO_CARD_TITLE_PENDING,
    HERO_CARD_TITLE_RANDOM,
    HERO_CREATION_PROMPT,
    HERO_CREATION_START_PROMPT,
    HERO_NOT_CREATED_ALERT,
    HERO_SETTING_FIRST_NOTE,
    SETTING_CONFIRM_DND,
    SETTING_CONFIRM_WARCRAFT,
    SETTING_MENU_TEXT,
    SETTING_SWITCHED_TEXT,
    WARCRAFT_CREATION_ADDENDUM,
    WARCRAFT_LORE_MISSING_NOTE,
    WARCRAFT_LORE_PROMPT,
    WARCRAFT_SETTING_INTRO,
    WARCRAFT_SETTING_OUTRO,
)

# ---------------------------------------------------------------------------
# 1. КОНФИГУРАЦИЯ
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# Загружаем переменные окружения из .env (если файла нет — берём из окружения ОС).
load_dotenv(ENV_PATH)

# Локальная база данных SQLite с листами персонажей и историей диалогов.
# На хостинге каталог проекта пересобирается при каждом деплое, поэтому файл базы
# держим ВНЕ образа: если задана переменная окружения DATA_DIR (например, путь к
# смонтированному volume), база хранится там и переживает пересборку контейнера;
# если DATA_DIR пуста — как и раньше, рядом с main.py (удобно локально).
DATA_DIR_ENV = os.getenv("DATA_DIR", "").strip()
if DATA_DIR_ENV:
    DATA_DIR_PATH = Path(DATA_DIR_ENV)
    DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
    DB_PATH = DATA_DIR_PATH / "bot_database.db"
else:
    DB_PATH = BASE_DIR / "bot_database.db"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# Ключ LLM (официальный API DeepSeek). Читаем ключ из переменной окружения
# DEEPSEEK_API_KEY; для совместимости принимаем также старые имена GEMINI_API_KEY
# и GROQ_API_KEY. Секреты в коде не храним: укажите ключ в файле .env
# (локально) или в переменных окружения сервера. Проверка наличия — в main().
LLM_API_KEY = (
    os.getenv("DEEPSEEK_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("GROQ_API_KEY")
    or ""
).strip()

# Официальный API DeepSeek (OpenAI-совместимый эндпоинт).
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"

# Надёжность запросов к LLM: таймаут и повторы при временных сбоях.
LLM_REQUEST_TIMEOUT = 60.0       # таймаут одного запроса к API, секунд
LLM_MAX_ATTEMPTS = 4             # всего попыток: 1 запрос + 3 повтора
LLM_RETRY_BASE_DELAY = 2.0       # базовая пауза между попытками, сек (удваивается)

MAX_HISTORY_MESSAGES = 20        # сколько последних сообщений держим в памяти и грузим из БД
DM_MAX_TOKENS = 1200             # лимит длины ответа Мастера
DM_TEMPERATURE = 0.65            # ниже 1.0 — меньше галлюцинаций и «псевдославянской» архаики в тексте
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
# 2.1 СЕТТИНГИ И БАЗА ЗНАНИЙ МИРОВ (choosing the world: D&D или Warcraft)
# ---------------------------------------------------------------------------

# Сеттинг партии: классический D&D (Забытые Королевства) или вселенная Warcraft
# (Азерот, эпоха Третьей Войны, 20–27 гг. ADP). Выбор хранится в листе персонажа
# (см. Character.setting) и влияет на системный промпт Мастера.
SETTING_DND_CLASSIC = "dnd_classic"
SETTING_WARCRAFT = "warcraft"
SETTING_CHOICES: tuple[str, ...] = (SETTING_DND_CLASSIC, SETTING_WARCRAFT)

# Префикс callback_data кнопок выбора сеттинга: «setting:dnd_classic» / «setting:warcraft»
# (укладывается в лимит Telegram в 64 байта).
SETTING_CALLBACK_PREFIX = "setting:"

# Человекочитаемые названия миров — для листа персонажа, подсказок и всплывающих сообщений.
SETTING_LABELS: dict[str, str] = {
    SETTING_DND_CLASSIC: "🎲 Классический D&D 2024 (Забытые Королевства)",
    SETTING_WARCRAFT: "⚔️ Вселенная Warcraft (Азерот, 20–27 гг. ADP)",
}


def normalize_setting(value: Any) -> str:
    """Приводит значение сеттинга к допустимому (при «мусоре» — классический D&D)."""
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in SETTING_CHOICES:
            return cleaned
    return SETTING_DND_CLASSIC


# ---------------------------------------------------------------------------
# 2.2 ОГРАНИЧЕНИЕ ДОСТУПА И ЧАСТОТЫ (белый список и антифлуд)
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    """Читает целое число из переменной окружения (при ошибке — значение по умолчанию)."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        logger.warning("Переменная %s=%r не является числом — беру %d.", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    """Читает число с плавающей точкой из окружения (при ошибке — значение по умолчанию)."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        logger.warning("Переменная %s=%r не является числом — беру %r.", name, raw, default)
        return default


def _parse_id_set(value: str) -> frozenset[int]:
    """Разбирает список Telegram user id из строки (через запятую или пробелы)."""
    ids: set[int] = set()
    for chunk in re.split(r"[\s,]+", value or ""):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.isdigit():
            ids.add(int(chunk))
        else:
            logger.warning("Игнорирую некорректный user id: %r.", chunk)
    return frozenset(ids)


# Белый список игроков (пусто — пускаем всех) и администраторы (обходят лимиты).
ALLOWED_USER_IDS = _parse_id_set(os.getenv("ALLOWED_USER_IDS", ""))
ADMIN_USER_IDS = _parse_id_set(os.getenv("ADMIN_USER_IDS", ""))


def _is_admin(user_id: int) -> bool:
    """True, если пользователь — администратор.

    Пустой ADMIN_USER_IDS трактуется как «все администраторы» (удобно в разработке);
    в продакшене список задаётся в .env, и тогда доступ к скрытым командам есть только у него.
    """
    return not ADMIN_USER_IDS or user_id in ADMIN_USER_IDS

# Ограничение частоты обращений на пользователя (скользящее окно).
RATE_LIMIT_MAX_REQUESTS = _env_int("RATE_LIMIT_MAX_REQUESTS", 20)
RATE_LIMIT_WINDOW_SECONDS = _env_float("RATE_LIMIT_WINDOW_SECONDS", 60.0)

# ---------------------------------------------------------------------------
# 3. СИСТЕМНЫЙ ПРОМПТ (DUNGEON MASTER)
# ---------------------------------------------------------------------------

# Тексты BASE_DM_PROMPT и CONTROL_BLOCK_INSTRUCTIONS перенесены в пакет prompts.

# Полный системный промпт: стиль Мастера + официальный справочник правил + служебный блок.
SYSTEM_PROMPT = "\n\n".join((BASE_DM_PROMPT, build_reference_digest(), CONTROL_BLOCK_INSTRUCTIONS))


@lru_cache(maxsize=32)
def build_system_prompt(
    setting: str = SETTING_DND_CLASSIC,
    *,
    species_name: Optional[str] = None,
    hero_ready: bool = False,
) -> str:
    """Собирает системный промпт Мастера под выбранный сеттинг и фазу игры.

    * "dnd_classic" — базовый промпт: стиль Мастера + справочник правил + служебный блок;
    * "warcraft"    — тот же промпт, но с блоком сеттинга, полной хроникой Азерота
      (WARCRAFT_LORE_PROMPT) и правилами игры в этом мире.

    :param species_name: вид (раса) героя; нужен, когда герой уже создан.
    :param hero_ready: True — герой создан (этап подтверждения или сама игра): справочник
        правил сокращается — без раздела «Создание персонажа» и с расовыми особенностями
        только текущего героя. False — этап создания героя: справочник полный.

    Функция кэшируется (lru_cache): большой текст хроники подмешивается один раз
    на сочетание «сеттинг + вид + фаза», а не пересобирается на каждый запрос к модели.
    """
    digest = build_reference_digest(full=not hero_ready, species_name=species_name)

    if normalize_setting(setting) != SETTING_WARCRAFT:
        return "\n\n".join((BASE_DM_PROMPT, digest, CONTROL_BLOCK_INSTRUCTIONS))

    lore = WARCRAFT_LORE_PROMPT or WARCRAFT_LORE_MISSING_NOTE
    return "\n\n".join(
        (
            BASE_DM_PROMPT,
            f"{WARCRAFT_SETTING_INTRO}\n{lore}",
            WARCRAFT_SETTING_OUTRO,
            digest,
            CONTROL_BLOCK_INSTRUCTIONS,
        )
    )


def setting_label(setting: str) -> str:
    """Человекочитаемое название сеттинга (для листа персонажа и сообщений игроку)."""
    return SETTING_LABELS[normalize_setting(setting)]

# ---------------------------------------------------------------------------
# 4. СТАТИЧНЫЕ ТЕКСТЫ И СЛУЖЕБНЫЕ СООБЩЕНИЯ
# ---------------------------------------------------------------------------

WELCOME_TEXT = (
    "🐉 Добро пожаловать за стол, искатель приключений!\n\n"
    "Я — твой Мастер Подземелий в духе Dungeons & Dragons 5e (2024). Я опишу мир, его "
    "опасности и судьбу твоего героя, но все решения остаются за тобой.\n\n"
    "Шаг 0 — выбери мир игры кнопками ниже:\n"
    "• 🎲 Забытые Королевства — классический D&D 2024.\n"
    "• ⚔️ Вселенная Warcraft (Азерот, 20–27 гг. ADP) — правила D&D 2024, но мир, фракции, "
    "города, монстры и NPC строго из хроники Warcraft.\n\n"
    "Шаг 1 — создай героя:\n"
    "• Назови имя, вид (раса), класс и пару слов о внешности или характере героя.\n"
    "• Виды по правилам PHB 2024: Человек, Эльф, Дварф, Гном, Полурослик, Драконорождённый, "
    "Тифлинг, Орк, Голиаф, Аасимар.\n"
    "• Классы: Воин, Варвар, Плут, Волшебник, Жрец, Следопыт, Паладин, Бард, Друид, Колдун, "
    "Монах, Чародей.\n"
    "• В сеттинге Warcraft вид героя — из народов Азерота (человек Штормграда или Терамора, "
    "дворф, гном, ночной эльф, орк, таурен, тролль, эльф крови, отрекшийся, дреней), а классы "
    "те же 12 из PHB 2024.\n"
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
    "• Играешь заклинателем — открывай раздел «📜 Заклинания» (/spells): там ячейки, применение\n"
    "  заклинаний, подготовка на день и отдых «🌙 Отдых» (/rest) для восстановления ячеек.\n"
    "Команды: /start, /reset, /hero, /roll <кубик>, /sheet, /inventory, /check, /spells, /rest.\n\n"
    "Предыдущая сессия сброшена. Сперва — герой, потом — приключение!"
)

# ---------------------------------------------------------------------------
# ЭТАП 1: ТЕКСТЫ СОЗДАНИЯ ПЕРСОНАЖА
# ---------------------------------------------------------------------------

# Фазы общения с игроком (см. creation_stage).
CREATION_STAGE_HERO = "creation"          # герой ещё не описан
CREATION_STAGE_CONFIRM = "confirmation"   # герой описан, ждём подтверждения игрока
CREATION_STAGE_PLAY = "play"              # герой подтверждён, идёт приключение

# Тексты фаз создания и выбора мира перенесены в пакет prompts (см. prompts/creation.py).

# Признаки того, что игрок просит сгенерировать героя вместо описания своего.
RANDOM_HERO_MARKERS: tuple[str, ...] = (
    "случайн", "наугад", "рандом", "random", "любой герой", "выбери за меня",
    "сгенерируй геро", "сгенерируй персон", "придумай за меня", "составь за меня",
)

# Признаки того, что игрок хочет свести счёты с жизнью: герой добровольно погибает.
# Срабатывание = нелепая гибель героя, удаление персонажа и перезапуск партии
# (см. _execute_character_selfdeath). Маркеры записаны без «ё»: текст игрока нормализуется.
SUICIDE_MARKERS: tuple[str, ...] = (
    "суицид", "самоубийств",
    "покончить с собой", "покончу с собой", "покончил с собой",
    "свести счет", "свести счеты", "сведу счет", "свожу счет", "сведение счетов",
    "покончить жизнь", "покончу жизнь", "лишить себя жизни", "лишу себя жизни",
    "убить себя", "убью себя", "убиваю себя", "убил себя",
    "зарезать себя", "зарежу себя", "зарезался",
    "вскрыть вены", "вскрою вены", "вскрыл вены", "перерезать вены", "перережу вены",
    "перерезать себе горло", "перережу себе горло",
    "повеситься", "повешусь", "повесился",
    "спрыгнуть с крыши", "спрыгну с крыши", "броситься с крыши", "брошусь с крыши",
    "броситься под поезд", "брошусь под поезд", "брошусь в пропасть",
    "застрелиться", "застрелюсь", "застрелился",
    "утопиться", "утоплюсь", "утопился",
    "сжечь себя", "сожгу себя", "поджечь себя",
    "выпить яд", "выпью яд", "принять яд", "отравиться", "отравлюсь",
    "пронзить себя", "пронжу себя", "воткнуть нож в себя", "воткну нож в себя",
    "не хочу жить", "хочу умереть", "хочу сдохнуть", "перестать жить",
)

# Ответ на дисклеймер о суициде. Только явное согласие в начале сообщения убивает героя;
# «нет» и любой другой ответ возвращают игрока в игру (см. handle_player_action).
SUICIDE_YES_PATTERN = re.compile(
    r"^\s*(?:да|yes|ага|угу|ок|окей|окэй|конечно|подтверждаю|подтвердить|"
    r"согласен|согласна|убивай|убивайте|приступай)\b",
    re.IGNORECASE,
)
SUICIDE_NO_PATTERN = re.compile(
    r"^\s*(?:нет|no|не\b|отмена|отменяю|отбой|передумал(?:а)?|стой|подожди|погоди|"
    r"жить|живу|продолжаем)",
    re.IGNORECASE,
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

# --- Добровольная гибель героя (суицид) ---
# Игрок решил свести счёты с жизнью: бот НЕ отговаривает — разыгрывает максимально нелепую и
# унизительную гибель героя, затем удаляет персонажа и запускает партию заново с чистого листа.
SUICIDE_RESTART_TEXT = (
    "🗑️ Персонаж стёрт из летописи: вместе с ним удалены его история, уровень, опыт, золото\n"
    "и снаряжение. Партия начинается заново — с чистого листа."
)

# Позорные эпилоги: детерминированная страховка на случай, если Мастер промолчит или начнёт
# читать мораль. «{name}» подставляет имя героя (см. build_suicide_epilogue).
SUICIDE_EPILOGUES: tuple[str, ...] = (
    "{name} решил уйти из жизни красиво. Не вышло: поскользнулся на мокром полу, снёс головой "
    "бадью с пойлом и захлебнулся — не водой, а позором. Стража пришла не спасать, а поржать.",

    "{name} торжественно вынес себе смертный приговор, перепутал петлю с верёвкой для белья и был "
    "насмерть придушен мокрым полотенцем. Палач, пришедший по расписанию, обиженно ушёл восвояси.",

    "{name} вонзил кинжал в грудь строго по учебнику и не рассчитал: клинок прошёл между рёбер, "
    "вылез сзади и пригвоздил к стене чужой кошель. Последним звуком в его жизни был хохот "
    "хозяина кошеля.",

    "{name} бросился с башни с гордым видом и застрял вниз головой в навесе для голубей, где "
    "вскоре и умер — от удушья и стыда одновременно. Птицы слетелись ещё заранее.",

    "{name} заказал «яд», но трактирщик перепутал и принёс тройную порцию несвежих потрохов. "
    "Итог тот же — только теперь это ещё и диарея со смертельным исходом.",

    "{name} заорал «Я ухожу навсегда!», схватил верёвку и повесился на ней в ближайшем колодце. "
    "Воду из этого колодца в городе с тех пор не пьют.",

    "{name} выпил мышьяк, тут же передумал, побежал искать лекаря, споткнулся и сломал шею о порог "
    "лазарета. Дверь как назло была закрыта на обеденный перерыв.",

    "{name} поджёг собственный плащ в знак отчаяния и вспомнил про запас алхимического масла "
    "поблизости. В историю он вошёл как «тот, кто устроил пожар на пустом месте и всё равно сдох».",
)

SUICIDE_EPILOGUE_FOOTER = (
    "\n\n━━━━━━━━━━━━━━━\n"
    "☠️ ГЕРОЙ МЁРТ. Окончательно, бесславно и очень смешно.\n\n"
)


def build_suicide_epilogue(character: Character) -> str:
    """Бесславный эпилог самоубийства героя (случайный шаблон + общий финал).

    Служит страховкой: гарантированно доводит до игрока нелепую гибель и факт удаления персонажа,
    даже если Мастер промолчал или отказался описывать сцену.
    """
    name = (character.name or DEFAULT_NAME).strip()
    body = random.choice(SUICIDE_EPILOGUES).format(name=name)
    return f"{body}{SUICIDE_EPILOGUE_FOOTER}"


# Дисклеймер перед добровольной гибелью героя: смерть — только по явному «да», иначе игра идёт дальше.
SUICIDE_CONFIRM_TEXT = (
    "⚠️ ДИСКЛЕЙМЕР.\n\n"
    "Похоже, ты хочешь свести счёты с жизнью — то есть убить своего персонажа.\n\n"
    "Если это так, напиши «ДА» — и герой погибнет прямо сейчас: окончательно, позорно и без "
    "воскрешения. На этом игра ЗАКОНЧИТСЯ, а персонаж будет СТЁРТ из летописи вместе с уровнем, "
    "опытом, золотом и снаряжением. Захочешь вернуться — начнёшь заново, с чистого листа.\n\n"
    "Если передумал — напиши «НЕТ», и история продолжится, будто ты ничего не говорил.\n\n"
    "Любой другой ответ я сочту за «нет»."
)

# Отказ от суицида: игрок вернулся в игру, персонаж жив.
SUICIDE_CANCEL_TEXT = (
    "✅ Ладно, живи пока. Ты остаёшься в игре — будто ничего и не было.\n\n"
    "Что ты делаешь?"
)

# Издевательства бота в момент гибели героя: игрок назван слабым и никчёмным (по запросу владельца).
SUICIDE_TAUNTS: tuple[str, ...] = (
    "Слабак. Даже уйти достойно не сумел — только развёл дешёвую драму.",
    "Никчёмный герой. Подвигов — ноль, а конец — как у последнего дурака.",
    "Жалкое зрелище. Мир даже не заметит, что тебя не стало. И правильно.",
    "Трус и бездарь. Ни одной славной истории — зато целая страница позора.",
    "Слаб духом и телом. Твой путь кончился ровно так, как ты его и вёл — никак.",
    "Никчёмность! Даже смерть у тебя вышла дешёвкой. В тавернах расскажут это как глупую шутку.",
    "Вот и всё, на что тебя хватило: сбежать от собственной жизни. Позор такому герою.",
)

# Тексты инлайн-клавиатур и «всплывающих» подсказок на кнопках.
ACTION_MENU_TEXT = "🎲 Быстрые действия: выбери бросок или нужный раздел."

CHECKS_MENU_TEXT = (
    "🧠 ПРОВЕРКИ И СПАСБРОСКИ ХАРАКТЕРИСТИК\n"
    "Нажми характеристику — я брошу d20, добавлю её модификатор из твоего листа "
    "и передам Мастеру готовый итог.\n"
    "Кнопки со 🛡 — спасброски: к характеристике прибавляется бонус мастерства, "
    "если класс владеет этим спасброском."
)

STALE_CALLBACK_TEXT = (
    "Эта кнопка устарела (сообщение слишком старое). "
    "Напиши новое действие в чат — под свежим ответом кнопки снова активны."
)

UNKNOWN_BUTTON_TEXT = "Неизвестная кнопка — попробуй ещё раз."

# Тексты раздела магии (карточка заклинаний, применение, подготовка и отдых).
SPELLS_MENU_TEXT = (
    "🪄 КНИГА ЗАКЛИНАНИЙ\n"
    "Применяй заклинания, готовь их на день или отдыхай, чтобы восстановить ячейки."
)

NOT_SPELLCASTER_TEXT = (
    "🪄 У этого героя нет магии: его класс не владеет заклинаниями.\n"
    "Магией пользуются Бард, Жрец, Друид, Паладин, Следопыт, Чародей, Колдун и Волшебник."
)

SPELL_MENU_STALE_TEXT = (
    "Меню заклинаний устарело. Открой его заново кнопкой «📜 Заклинания»."
)

SPELL_NOT_AVAILABLE_ALERT = (
    "Это заклинание сейчас недоступно: его нет в списке заговоров или оно не "
    "заготовлено на сегодня."
)

CAST_MENU_TEXT = (
    "🔥 ЧТО ПРИМЕНИТЬ\n"
    "Нажми заклинание — код спишет ячейку нужного круга и передаст применение Мастеру. "
    "Заговоры ячеек не тратят."
)

PREP_MENU_TEXT = (
    "⚡ ПОДГОТОВКА ЗАКЛИНАНИЙ\n"
    "Нажми заклинание, чтобы заготовить или снять его: ✅ — готово к применению, "
    "❌ — не заготовлено."
)

SPONTANEOUS_PREP_ALERT = (
    "Твой класс — спонтанный заклинатель: все изученные заклинания всегда готовы к "
    "применению, менять список подготовки не нужно."
)

REST_MENU_TEXT = (
    "🌙 ОТДЫХ\n"
    "Продолжительный отдых (8 часов) восстанавливает ВСЕ ячейки заклинаний.\n"
    "Короткий отдых (1 час) восстанавливает «магию пакта» Колдуна."
)

NO_SLOTS_ALERT = "У этого героя нет ячеек заклинаний."

REST_NOT_CASTERTEXT = (
    "У этого героя нет магии, поэтому ячейки заклинаний восстанавливать нечего.\n"
    "Отдохнуть всё равно можно — просто опиши отдых словами."
)


API_ERROR_TEXT = (
    "⚠️ Мастер ненадолго отвлёкся: не удалось связаться с оракулом.\n"
    "Попробуй повторить сообщение через несколько секунд."
)

GENERIC_ERROR_TEXT = (
    "⚠️ Что-то пошло не так при обработке твоего действия. Попробуй ещё раз."
)

# Сообщения системы доступа: белый список и превышение лимита обращений.
ACCESS_DENIED_TEXT = (
    "🔒 Извини, бот закрыт для посторонних. "
    "Попроси владельца добавить твой Telegram ID в список доступа."
)

RATE_LIMIT_TEXT = (
    "⏳ Слишком много сообщений подряд. "
    "Подожди несколько секунд и повтори — так Мастер успеет ответить как следует."
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

        Математика броска уже посчитана кодом, поэтому Мастеру передаётся неоспоримое
        равенство «кубик + мод = ИТОГ»: он не должен пересчитывать модификаторы, метать
        кубик заново или подменять итог.

        :param purpose: зачем бросали (например, «бросок урона 1d8+3 (Длинный меч)»).
        """
        dice_total = sum(self.kept)
        if self.is_flat:
            # Кубики не бросались: показываем это явно (в зачёт идёт только модификатор).
            detail = " (без броска кубиков, фиксированное значение)"
        elif self.mode and len(self.rolls) > 1:
            detail = f" (d20: {self.rolls[0]}/{self.rolls[1]}, в зачёт {self.kept[0]})"
        elif len(self.rolls) > 1:
            detail = f" ({self.count}d{self.sides}: {', '.join(map(str, self.rolls))})"
        else:
            detail = ""
        reason = f" Повод: {purpose}." if purpose else ""
        return (
            f"[СИСТЕМА]: Игрок выбросил на кубике {dice_total} + мод {self.modifier:+d} "
            f"= ИТОГ {self.total}{detail}.{reason} {DM_ROLL_INSTRUCTION}"
        )


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

# Краткая сводка локации (HUD): где герой и какова его цель, пока Мастер не сменил их.
DEFAULT_LOCATION = "Неизвестно"
DEFAULT_QUEST = "Исследовать местность"

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


def _unique_spell_list(value: Any) -> list[str]:
    """Приводит значение к списку названий заклинаний без пустых строк и дублей."""
    items: list[str] = []
    seen: set[str] = set()
    for candidate in _as_text_list(value):
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(candidate)
    return items


def _normalize_ability_codes(value: Any) -> list[str]:
    """Приводит значение к списку кодов характеристик ('str', 'dex', ...) без дублей."""
    codes: list[str] = []
    seen: set[str] = set()
    for candidate in _as_text_list(value):
        code = normalize_ability_key(candidate)
        if code is None and candidate.lower() in ABILITIES:
            code = candidate.lower()
        if code is not None and code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


# Извлекает первое число из строки вида «+50 XP», «-10 hp», «15».
_SIGNED_INT_IN_TEXT = re.compile(r"[-+]?\d+")


def _as_change_int(value: Any) -> int:
    """Приводит изменение листа (XP/HP/gp) к int, терпимо к строкам «+50 XP» / «-10 hp».

    Отличие от :func:`_as_int`: если модель прислала строку с числом и подписью
    (например, ``"получено 25 XP"``), из неё извлекается именно число.
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        match = _SIGNED_INT_IN_TEXT.search(value)
        if match is not None:
            return int(match.group())
    return 0


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


def _as_optional_label(value: Any) -> Optional[str]:
    """Возвращает непустую текстовую метку (локация, цель) или None.

    В отличие от `_as_optional_text`, числа и прочий «мусор» меткой не считаются:
    сводка HUD должна быть осмысленной строкой, иначе берётся значение по умолчанию.
    """
    if isinstance(value, str) and value.strip():
        return value.strip()
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
    # Краткая сводка локации (HUD): где герой сейчас и какова его текущая цель.
    # Обновляется служебным JSON-блоком Мастера при смене обстановки (см. apply_control).
    location: str = DEFAULT_LOCATION
    quest: str = DEFAULT_QUEST
    # Сеттинг партии: "dnd_classic" (Забытые Королевства) или "warcraft" (Азерот).
    # Выбирается игроком в начале создания персонажа (кнопки SETTINGS_KEYBOARD),
    # хранится в базе и определяет, подмешивать ли в промпт Мастера хронику Warcraft.
    setting: str = SETTING_DND_CLASSIC
    # --- Магия (spellcasting) ---
    # Поля магии заполняются автоматически по классу и уровню (см. _normalize_spellcasting),
    # поэтому у записей прежних версий они просто инициализируются значениями по умолчанию.
    is_spellcaster: bool = False
    # Код характеристики магии класса: 'int', 'wis' или 'cha' (пусто у не-заклинателей).
    spellcasting_ability: str = ""
    # Ячейки заклинаний: {'1': {'total': 4, 'current': 2}, ...} — круг -> ячейки.
    spell_slots: dict[str, dict[str, int]] = field(default_factory=dict)
    # Заговоры (0 круг) — ячеек не тратят и в лимит заготовки не входят.
    cantrips: list[str] = field(default_factory=list)
    # Все изученные заклинания 1+ круга.
    spells_known: list[str] = field(default_factory=list)
    # Заготовленные на сегодня заклинания (у спонтанных кастеров = spells_known).
    spells_prepared: list[str] = field(default_factory=list)
    # --- Владения (proficiencies) ---
    # Коды характеристик, по которым класс даёт владение спасбросками: 'str', 'con' и т.п.
    # Используются при расчёте модификатора спасброска (см. save_modifier).
    save_proficiencies: list[str] = field(default_factory=list)
    # Названия навыков (см. SKILLS в справочнике), которыми владеет герой.
    skill_proficiencies: list[str] = field(default_factory=list)
    # Оружие/категории оружия, которыми владеет герой (для атак и урона).
    weapon_proficiencies: list[str] = field(default_factory=list)

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
        # Сводка локации и цели всегда непустая: пустая строка означает «не задано».
        self.location = (self.location or "").strip()[:120] or DEFAULT_LOCATION
        self.quest = (self.quest or "").strip()[:120] or DEFAULT_QUEST
        # Сеттинг: любое неизвестное значение трактуем как классический D&D.
        self.setting = normalize_setting(self.setting)
        # Владения: чистим списки и подтягиваем владения спасбросками из класса (PHB 2024).
        self.save_proficiencies = _normalize_ability_codes(self.save_proficiencies)
        self.skill_proficiencies = _unique_spell_list(self.skill_proficiencies)
        self.weapon_proficiencies = _unique_spell_list(self.weapon_proficiencies)
        self._sync_save_proficiencies()
        # Магия: ячейки, заговоры и заклинания приводим к правилам класса и уровня.
        self._normalize_spellcasting()

    def _normalize_spellcasting(self) -> None:
        """Инициализирует поля магии по классу и уровню (безопасно для записей прежних версий).

        Если у класса есть магия, а списков ещё нет (герой создан до появления системы
        заклинаний), они заполняются классовыми заговорами и стартовыми заклинаниями
        1-го уровня. Текущий остаток ячеек из базы бережно сохраняется.
        """
        ability = spellcasting_ability_for_class(self.class_name)
        if ability is None:
            self.is_spellcaster = False
            self.spellcasting_ability = ""
            self.spell_slots = {}
            self.cantrips = []
            self.spells_known = []
            self.spells_prepared = []
            return

        self.is_spellcaster = True
        self.spellcasting_ability = ability

        # Количество ячеек берём из таблиц правил, остаток (current) — из базы, если он есть.
        totals = spell_slots_for_level(self.class_name, self.level)
        raw_slots = self.spell_slots if isinstance(self.spell_slots, Mapping) else {}
        normalized_slots: dict[str, dict[str, int]] = {}
        for circle, total in totals.items():
            stored = raw_slots.get(circle)
            current = total
            if isinstance(stored, Mapping):
                current = max(0, min(_as_int(stored.get("current", total)), total))
            normalized_slots[circle] = {"total": total, "current": current}
        self.spell_slots = normalized_slots

        self.cantrips = _unique_spell_list(self.cantrips)
        self.spells_known = _unique_spell_list(self.spells_known)
        self.spells_prepared = _unique_spell_list(self.spells_prepared)

        # Заговоры и заклинания 1+ круга — непересекающиеся списки.
        cantrip_set = set(self.cantrips)
        self.spells_known = [name for name in self.spells_known if name not in cantrip_set]
        self.spells_prepared = [name for name in self.spells_prepared if name not in cantrip_set]

        if not self.cantrips:
            self.cantrips = list(default_cantrips_for_class(self.class_name))
        if not self.spells_known:
            self.spells_known = list(default_known_spells_for_class(self.class_name))

        if is_spontaneous_caster(self.class_name):
            # Спонтанные заклинатели всегда держат готовыми всё изученное.
            self.spells_prepared = list(self.spells_known)
            return

        # Подготовленными могут быть только изученные заклинания, но не больше лимита.
        prepared = [name for name in self.spells_prepared if name in self.spells_known]
        limit = self.max_prepared
        if not prepared:
            prepared = list(self.spells_known[:limit])
        self.spells_prepared = prepared[:limit]

    def init_default_spellcasting(self) -> None:
        """Заполняет магию «по умолчанию» для нового героя (класс и характеристики заданы).

        Вызывается кодом при создании персонажа (см. apply_starter_loadout), когда
        характеристики уже выставлены по стандартному набору и лимит подготовки известен.
        """
        if not self.is_spellcaster:
            return
        self.cantrips = list(default_cantrips_for_class(self.class_name))
        self.spells_known = list(default_known_spells_for_class(self.class_name))
        if is_spontaneous_caster(self.class_name):
            self.spells_prepared = list(self.spells_known)
        else:
            self.spells_prepared = list(self.spells_known[: self.max_prepared])

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

    def roll_modifier(self, code: str, *, proficient: bool = False) -> int:
        """Модификатор броска d20: характеристика (+ бонус мастерства при владении).

        Единая точка расчёта модификаторов для проверок, спасбросков и атак.
        Считается только кодом — Мастер не должен пересчитывать его «на глаз».
        """
        modifier = self.ability_mod(code)
        if proficient:
            modifier += self.proficiency_bonus
        return modifier

    def save_modifier(self, code: str) -> int:
        """Модификатор спасброска: характеристика + бонус мастерства при владении классом."""
        return self.roll_modifier(code, proficient=code in self.save_proficiencies)

    def skill_modifier(self, skill_name: str) -> int:
        """Модификатор проверки навыка: характеристика навыка + бонус мастерства при владении."""
        ability = SKILLS.get(skill_name)
        if ability is None:
            return 0
        owned = {name.strip().lower() for name in self.skill_proficiencies}
        return self.roll_modifier(ability, proficient=skill_name.strip().lower() in owned)

    def _sync_save_proficiencies(self, force: bool = False) -> None:
        """Заполняет владения спасбросками по классу (PHB 2024).

        Владения спасбросками фиксированы правилами класса, поэтому если список ещё не
        задан (или ``force=True`` при смене класса), он берётся из справочника классов.
        """
        if self.save_proficiencies and not force:
            return
        key = resolve_class_key(self.class_name)
        if key is None:
            return
        self.save_proficiencies = [
            code for code in CLASSES[key].get("saves", ()) if code in ABILITIES
        ]

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

    # --- Магия: производные величины и операции с ячейками ---

    @property
    def spell_save_dc(self) -> int:
        """КС спасброска от заклинаний: 8 + бонус мастерства + модификатор характеристики."""
        if not self.is_spellcaster:
            return 0
        return 8 + self.proficiency_bonus + self.ability_mod(self.spellcasting_ability)

    @property
    def spell_attack_bonus(self) -> int:
        """Модификатор атаки заклинанием: бонус мастерства + модификатор характеристики."""
        if not self.is_spellcaster:
            return 0
        return self.proficiency_bonus + self.ability_mod(self.spellcasting_ability)

    @property
    def max_prepared(self) -> int:
        """Лимит заготовленных заклинаний (уровень + модификатор характеристики, мин. 1).

        Для спонтанных заклинателей лимита нет — возвращается число изученных заклинаний.
        Для класса без магии возвращается 0.
        """
        if not self.is_spellcaster:
            return 0
        limit = max_prepared_spells(
            self.class_name, self.level, self.ability_mod(self.spellcasting_ability or "int")
        )
        if limit is None:
            return len(self.spells_known)
        return max(1, limit)

    @property
    def total_slots(self) -> int:
        """Суммарное число ячеек заклинаний всех кругов."""
        return sum(slot["total"] for slot in self.spell_slots.values())

    @property
    def available_slots(self) -> int:
        """Сколько ячеек заклинаний сейчас свободно."""
        return sum(slot["current"] for slot in self.spell_slots.values())

    def spell_circle(self, spell_name: str) -> int:
        """Круг заклинания: 0 — заговор, иначе круг из справочника (по умолчанию 1)."""
        if spell_name in self.cantrips:
            return 0
        level = spell_level(spell_name)
        return 1 if level is None else level

    def castable_spells(self) -> list[tuple[str, int]]:
        """Доступные к применению заклинания: (название, круг); заговоры идут первыми."""
        entries: list[tuple[str, int]] = [(name, 0) for name in self.cantrips]
        entries += [(name, self.spell_circle(name)) for name in self.spells_prepared]
        return entries

    def can_prepare_more(self) -> bool:
        """Есть ли ещё место в лимите подготовки заклинаний."""
        return len(self.spells_prepared) < self.max_prepared

    def toggle_prepared(self, spell_name: str) -> bool:
        """Переключает заготовку заклинания. True — заготовлено, False — снято."""
        if spell_name in self.spells_prepared:
            self.spells_prepared = [name for name in self.spells_prepared if name != spell_name]
            return False
        self.spells_prepared.append(spell_name)
        return True

    def cast_spell(self, spell_name: str) -> "SpellCastResult":
        """Списывает ячейку круга при применении заклинания (заговоры ячеек не тратят).

        Применить можно только заговор или заготовленное заклинание: неизученные и
        снятые с подготовки заклинания отклоняются с подсказкой.

        :return: результат с заметкой игроку и служебным сообщением для Мастера; при
            нехватке ячеек или недоступном круге ``ok=False`` и заполнено ``alert``.
        """
        if spell_name not in self.cantrips and spell_name not in self.spells_prepared:
            return SpellCastResult(ok=False, alert=SPELL_NOT_AVAILABLE_ALERT)

        circle = self.spell_circle(spell_name)
        circle_label = SPELL_LEVEL_RU.get(circle, f"{circle} круг")

        if circle == 0:
            return SpellCastResult(
                ok=True,
                note=f"🪄 Ты применяешь заговор «{spell_name}» (ячейки не тратятся).",
                context=(
                    f"[СИСТЕМА] Игрок применяет заклинание '{spell_name}' "
                    f"(заговор, ячейки не требуются)."
                ),
            )

        slot = self.spell_slots.get(str(circle))
        if not slot or slot["total"] <= 0:
            return SpellCastResult(
                ok=False,
                alert=f"Заклинания {circle_label} тебе пока недоступны.",
            )
        if slot["current"] <= 0:
            return SpellCastResult(
                ok=False,
                alert=f"🔒 Нет свободных ячеек {circle_label}! Отдохни, чтобы восстановить их.",
            )

        slot["current"] -= 1
        left, total = slot["current"], slot["total"]
        return SpellCastResult(
            ok=True,
            note=(
                f"🪄 Ты применяешь «{spell_name}» ({circle_label}). "
                f"Осталось ячеек {circle_label}: {left}/{total}."
            ),
            context=(
                f"[СИСТЕМА] Игрок применяет заклинание '{spell_name}' ({circle_label}). "
                f"Осталось ячеек {circle_label}: {left}/{total}."
            ),
        )

    def restore_spell_slots(self, long_rest: bool) -> list[str]:
        """Восстанавливает ячейки при отдыхе. Возвращает заметки для игрока.

        Продолжительный (длинный) отдых возвращает все ячейки; короткий — только «магию
        пакта» Колдуна (у остальных классов на коротком отдыхе ячейки не восстанавливаются).
        """
        if not self.spell_slots:
            return []
        if not (long_rest or is_pact_caster(self.class_name)):
            return []

        notes: list[str] = []
        for circle, slot in self.spell_slots.items():
            slot["current"] = slot["total"]
            label = SPELL_LEVEL_RU.get(_as_int(circle), f"{circle} круг")
            notes.append(f"🔋 {label}: {slot['current']}/{slot['total']}.")
        return notes

    # --- Изменение листа персонажа ---

    def _level_up(self) -> str:
        """Повышает уровень: увеличивает максимум HP и лечит на ту же величину."""
        self.level += 1
        gained = average_hit_points(self.hit_die, self.ability_mod("con"))
        self.max_hp += gained
        self.current_hp = min(self.max_hp, self.current_hp + gained)
        # Новый уровень открывает новые ячейки заклинаний (текущий остаток сохраняется).
        self._normalize_spellcasting()
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
                # Смена класса полностью пересобирает магию: списки заклинаний и ячейки.
                self.cantrips = []
                self.spells_known = []
                self.spells_prepared = []
                self.spell_slots = {}
                self._normalize_spellcasting()
                # Владения спасбросками — свойство класса: пересчитываем при его смене.
                self._sync_save_proficiencies(force=True)

        description = data.get("description")
        if isinstance(description, str) and description.strip():
            cleaned_description = description.strip()[:400]
            if cleaned_description != self.description:
                self.description = cleaned_description
                notes.append("🖋️ Описание героя записано.")

        # Сводка локации (HUD): Мастер передаёт её при смене обстановки.
        location = data.get("location")
        if isinstance(location, str) and location.strip():
            cleaned_location = location.strip()[:120]
            if cleaned_location != self.location:
                self.location = cleaned_location
                notes.append(f"📍 Новая локация: {self.location}.")

        # Текущая цель героя — вторая строка сводки.
        quest = data.get("quest")
        if isinstance(quest, str) and quest.strip():
            cleaned_quest = quest.strip()[:120]
            if cleaned_quest != self.quest:
                self.quest = cleaned_quest
                notes.append(f"🎯 Новая цель: {self.quest}.")

        level = _as_int(data.get("level"))
        if 1 <= level <= MAX_LEVEL and level != self.level:
            self.level = level
            notes.append(f"🎖️ Уровень: {self.level}.")
            # Уровень определяет число ячеек заклинаний — пересчитываем их.
            self._normalize_spellcasting()

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
                # Модификатор характеристики магии задаёт лимит подготовки — подрезаем список,
                # если характеристику понизили через служебный блок.
                if self.is_spellcaster and not is_spontaneous_caster(self.class_name):
                    self.spells_prepared = self.spells_prepared[: self.max_prepared]

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

        xp_gained = _as_change_int(data.get("xp_gained"))
        if xp_gained > 0:
            self.xp += xp_gained
            notes.append(f"✨ Получено {xp_gained} XP (всего {self.xp}).")
            notes.extend(self._apply_level_ups())

        hp_change = _as_change_int(data.get("hp_change"))
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

        gp_change = _as_change_int(data.get("gp_change"))
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
            "location": self.location,
            "quest": self.quest,
            "setting": normalize_setting(self.setting),
            "is_spellcaster": bool(self.is_spellcaster),
            "spellcasting_ability": self.spellcasting_ability,
            "spell_slots": {
                str(circle): {"total": int(slot["total"]), "current": int(slot["current"])}
                for circle, slot in self.spell_slots.items()
            },
            "cantrips": list(self.cantrips),
            "spells_known": list(self.spells_known),
            "spells_prepared": list(self.spells_prepared),
            "save_proficiencies": list(self.save_proficiencies),
            "skill_proficiencies": list(self.skill_proficiencies),
            "weapon_proficiencies": list(self.weapon_proficiencies),
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

        raw_slots = data.get("spell_slots")
        spell_slots: dict[str, dict[str, int]] = {}
        if isinstance(raw_slots, Mapping):
            for raw_circle, raw_slot in raw_slots.items():
                circle = str(raw_circle).strip()
                if not circle or not isinstance(raw_slot, Mapping):
                    continue
                spell_slots[circle] = {
                    "total": _as_int(raw_slot.get("total")),
                    "current": _as_int(raw_slot.get("current")),
                }

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
            location=_as_optional_label(data.get("location")) or DEFAULT_LOCATION,
            quest=_as_optional_label(data.get("quest")) or DEFAULT_QUEST,
            setting=normalize_setting(data.get("setting")),
            is_spellcaster=bool(data.get("is_spellcaster")),
            spellcasting_ability=_as_optional_text(data.get("spellcasting_ability")) or "",
            spell_slots=spell_slots,
            cantrips=_unique_spell_list(data.get("cantrips")),
            spells_known=_unique_spell_list(data.get("spells_known")),
            spells_prepared=_unique_spell_list(data.get("spells_prepared")),
            save_proficiencies=_normalize_ability_codes(data.get("save_proficiencies")),
            skill_proficiencies=_unique_spell_list(data.get("skill_proficiencies")),
            weapon_proficiencies=_unique_spell_list(data.get("weapon_proficiencies")),
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


@dataclass
class SpellCastResult:
    """Результат применения заклинания кодом бота (см. Character.cast_spell).

    :param ok: успешно ли применение (ячейка списана или это заговор).
    :param note: текст-подтверждение для игрока (при успехе).
    :param context: служебное сообщение Мастеру об использованном заклинании.
    :param alert: всплывающая подсказка кнопки при отказе (не хватает ячеек и т.п.).
    """

    ok: bool
    note: str = ""
    context: str = ""
    alert: str = ""


# ---------------------------------------------------------------------------
# Форматирование листа персонажа и разбор служебного блока Мастера
# ---------------------------------------------------------------------------


def hp_progress_bar(current_hp: int, max_hp: int, width: int = 20) -> str:
    """Рисует прогресс-бар HP, например: ████████░░░░░░░░░░░░"""
    ratio = 0.0 if max_hp <= 0 else max(0.0, min(1.0, current_hp / max_hp))
    filled = round(ratio * width)
    return "█" * filled + "░" * (width - filled)


def format_hud(character: Character) -> str:
    """Краткая сводка локации и цели (HUD), которая идёт перед ответами Мастера в игре."""
    return (
        f"📍 Локация: {character.location}\n"
        f"🎯 Текущая цель: {character.quest}"
    )


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

    # Расовые особенности (черты вида) из справочника правил — если вид распознан.
    race_traits = species_traits(character.race)
    traits_lines: list[str] = (
        ["", "🧬 Черты (расовые)", *(f"• {trait}" for trait in race_traits)]
        if race_traits
        else []
    )

    # Магический блок в листе показываем только заклинателям.
    magic_lines: list[str] = []
    if character.is_spellcaster:
        ability = ABILITY_FULL_RU.get(
            character.spellcasting_ability, character.spellcasting_ability.upper()
        )
        magic_lines = [
            "",
            f"🪄 Магия класса ({ability}): КС спасброска {character.spell_save_dc}, "
            f"атака заклинанием {format_modifier(character.spell_attack_bonus)}",
            *format_slots_tracker(character.spell_slots),
            f"🌟 Заговоры: {', '.join(character.cantrips) or 'нет'}",
            f"📚 Готово ({len(character.spells_prepared)}/{character.max_prepared}): "
            f"{', '.join(character.spells_prepared) or 'нет'}",
        ]

    return "\n".join(
        [
            "📜 ЛИСТ ПЕРСОНАЖА",
            "",
            format_hud(character),
            "",
            f"📛 Имя: {character.name}",
            f"🧬 Раса: {character.race}",
            f"⚔️ Класс: {character.class_name} (кость хитов d{character.hit_die})",
            f"🌍 Мир: {setting_label(character.setting)}",
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
            *magic_lines,
            *traits_lines,
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


def _spell_circle_label(circle: int) -> str:
    """Русское название круга заклинания для карточек и кнопок."""
    return SPELL_LEVEL_RU.get(circle, f"{circle} круг")


def _spell_circle_locked(character: Character, circle: int) -> bool:
    """Закончились ли ячейки нужного круга (заговоры не «запираются» никогда)."""
    if circle <= 0:
        return False
    slot = character.spell_slots.get(str(circle))
    return slot is None or slot["current"] <= 0


def format_slots_tracker(spell_slots: Mapping[str, Mapping[str, int]]) -> list[str]:
    """Строки трекера ячеек заклинаний: «🔋 1 круг: [████░░░░] 2/4»."""
    if not spell_slots:
        return ["🔋 Ячейки: нет — заговоры ячеек не тратят."]
    lines: list[str] = []
    for circle in sorted(spell_slots, key=int):
        slot = spell_slots[circle]
        total = int(slot.get("total", 0))
        current = int(slot.get("current", 0))
        bar = hp_progress_bar(current, total, width=12)
        lines.append(f"🔋 {_spell_circle_label(int(circle))}: [{bar}] {current}/{total}")
    return lines


def format_spells_card(character: Character) -> str:
    """Карточка «📜 Книга заклинаний»: характеристика, КС, ячейки и заговоры."""
    if not character.is_spellcaster:
        return NOT_SPELLCASTER_TEXT

    ability = ABILITY_FULL_RU.get(
        character.spellcasting_ability, character.spellcasting_ability.upper()
    )
    lines = [
        SPELLS_MENU_TEXT,
        "",
        format_hud(character),
        "",
        f"🧠 Магия класса: {ability}",
        f"🎯 КС спасброска от заклинаний: {character.spell_save_dc} | "
        f"✨ Атака заклинанием: {format_modifier(character.spell_attack_bonus)}",
        "",
        "📖 Ячейки заклинаний",
        *format_slots_tracker(character.spell_slots),
        "",
        f"🌟 Заговоры ({len(character.cantrips)})",
    ]
    if character.cantrips:
        lines.extend(f" • {name}" for name in character.cantrips)
    else:
        lines.append(" • (нет)")

    lines.append("")
    if is_spontaneous_caster(character.class_name):
        lines.append(
            f"📚 Изученные заклинания ({len(character.spells_known)}), "
            "все готовы к применению"
        )
    else:
        lines.append(
            f"📚 Готовые заклинания ({len(character.spells_prepared)}/"
            f"{character.max_prepared})"
        )

    if character.spells_prepared:
        lines.extend(
            f" • {name} ({_spell_circle_label(character.spell_circle(name))})"
            for name in character.spells_prepared
        )
    elif is_spontaneous_caster(character.class_name):
        lines.append(" • (нет изученных заклинаний)")
    else:
        lines.append(" • (нет — заготовь их в разделе «⚡ Подготовка»)")
    return "\n".join(lines)


def format_cast_menu(character: Character) -> str:
    """Карточка «🔥 Что применить»: трекер ячеек и список доступных заклинаний."""
    lines = [
        CAST_MENU_TEXT,
        "",
        format_hud(character),
        "",
        "📖 Ячейки заклинаний",
        *format_slots_tracker(character.spell_slots),
        "",
    ]
    if character.available_slots == 0:
        # Ячейки кончились: объясняем значок 🔒 до списка заклинаний.
        lines.append(NO_SLOTS_ALERT)
    lines.append("🪄 Доступно сейчас:")
    spells = character.castable_spells()
    if not spells:
        lines.append(" • (нечего применять)")
    for name, circle in spells:
        mark = "🔒" if _spell_circle_locked(character, circle) else "•"
        lines.append(f" {mark} {name} ({_spell_circle_label(circle)})")
    return "\n".join(lines)


def format_prep_menu(character: Character) -> str:
    """Карточка «⚡ Подготовка заклинаний»: изученные заклинания с метками готовности."""
    if not character.is_spellcaster:
        return NOT_SPELLCASTER_TEXT

    if is_spontaneous_caster(character.class_name):
        header = ["⚡ ПОДГОТОВКА ЗАКЛИНАНИЙ", "", SPONTANEOUS_PREP_ALERT]
    else:
        header = [
            PREP_MENU_TEXT,
            "",
            f"Заготовлено {len(character.spells_prepared)} из {character.max_prepared} "
            f"(уровень {character.level} + модификатор характеристики).",
        ]

    lines = [*header, "", "📚 Изученные заклинания"]
    if character.spells_known:
        for name in character.spells_known:
            mark = "✅" if name in character.spells_prepared else "❌"
            lines.append(f" {mark} {name} ({_spell_circle_label(character.spell_circle(name))})")
    else:
        lines.append(" • (нет)")
    return "\n".join(lines)


def format_rest_menu(character: Character) -> str:
    """Карточка «🌙 Отдых»: что восстановит короткий и продолжительный отдых."""
    lines = [
        REST_MENU_TEXT,
        "",
        f"❤️ HP: {character.current_hp}/{character.max_hp} "
        f"[{hp_progress_bar(character.current_hp, character.max_hp)}]",
        "",
    ]
    if character.is_spellcaster:
        lines.append("📖 Ячейки сейчас")
        lines.extend(format_slots_tracker(character.spell_slots))
        if is_pact_caster(character.class_name):
            lines.append("☝️ Колдун: короткий отдых тоже восстанавливает магию пакта.")
        else:
            lines.append("☝️ На коротком отдыхе ячейки не восстанавливаются.")
    else:
        lines.append(REST_NOT_CASTERTEXT)
    return "\n".join(lines)


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
        f"сеттинг: {setting_label(character.setting)}",
        f"уровень: {character.level}",
        f"HP: {character.current_hp}/{character.max_hp}",
        f"КД: {character.armor_class}",
        f"характеристики: {abilities}",
    ]
    if character.description:
        parts.append(f"описание: {character.description}")
    if character.inventory:
        parts.append("снаряжение: " + ", ".join(character.inventory))
    if character.is_spellcaster:
        # Состояние магии: Мастеру важно знать ячейки и готовые заклинания прямо в промпте.
        slots = ", ".join(
            f"{_spell_circle_label(int(circle))} {slot['current']}/{slot['total']}"
            for circle, slot in sorted(character.spell_slots.items(), key=lambda pair: int(pair[0]))
        )
        parts.append(
            f"магия: КС спасброска {character.spell_save_dc}, атака заклинанием "
            f"{format_modifier(character.spell_attack_bonus)}, ячейки: {slots or 'нет'}; "
            f"заговоры: {', '.join(character.cantrips) or 'нет'}; "
            f"готовые заклинания: {', '.join(character.spells_prepared) or 'нет'}"
        )
    return "Данные героя — " + "; ".join(parts) + "."


def creation_prompt(session: Session) -> str:
    """Инструкция Мастеру в фазе создания: первое сообщение игры или продолжение.

    Для сеттинга Warcraft к инструкции добавляется подсказка о видах Азерота и
    классах PHB 2024 (см. WARCRAFT_CREATION_ADDENDUM).
    """
    prompt = HERO_CREATION_START_PROMPT if len(session.history) <= 1 else HERO_CREATION_PROMPT
    if normalize_setting(session.character.setting) == SETTING_WARCRAFT:
        return f"{prompt}\n{WARCRAFT_CREATION_ADDENDUM}"
    return prompt


def hero_confirmation_prompt(character: Character) -> str:
    """Инструкция Мастеру: герой создан, но игрок его ещё не подтвердил."""
    prompt = (
        "[СИСТЕМА] Этап создания персонажа: герой записан в лист, но игрок его ещё НЕ подтвердил. "
        "Приключение и пролог начинать ЗАПРЕЩЕНО.\n"
        f"{hero_summary(character)}\n"
        "Задание: коротко (1–2 абзаца) отреагируй на сообщение игрока. Просит изменить героя — "
        "исправь нужные поля (\"name\", \"race\", \"class_name\", \"description\") в служебном "
        "JSON-блоке; описывает действия вместо героя — вежливо напомни, что сначала нужно "
        "подтвердить героя. В конце спроси, всё ли верно с героем: подтвердить можно словом «да» "
        "или кнопкой «✅ Подтвердить героя». Пролог не начинай."
    )
    if normalize_setting(character.setting) == SETTING_WARCRAFT:
        return f"{prompt}\n{WARCRAFT_CREATION_ADDENDUM}"
    return prompt


def prologue_prompt(character: Character) -> str:
    """Инструкция Мастеру начать вводную сцену после подтверждения героя."""
    if normalize_setting(character.setting) == SETTING_WARCRAFT:
        # В Азероте мир и завязка берутся из хроники — «придумывать название мира» нельзя.
        scene = (
            "начни вводную сцену пролога в мире Азерота (20–27 гг. ADP): атмосферно опиши, где и "
            "как начинается путь героя, бери место, время и обстановку из хроники Warcraft, покажи "
            "одну из сил эпохи (Альянс, Орда, Плеть, Культ Проклятых) и придумай короткую завязку "
            "в духе тёмного героического фэнтези, дай одну-две зацепки и остановись в точке выбора."
        )
    else:
        scene = (
            "начни вводную сцену пролога по правилам D&D 2024: атмосферно опиши, где и как "
            "начинается путь героя, придумай название мира и короткую завязку в духе тёмного "
            "героического фэнтези, дай одну-две зацепки и остановись в точке выбора."
        )
    return (
        "[СИСТЕМА] Игрок подтвердил героя. Этап создания персонажа завершён — начинается игра.\n"
        f"{hero_summary(character)}\n"
        f"Задание: {scene} Назови героя по имени («{character.name}, твоя история начинается…»). "
        "НЕ описывай действия, слова и мысли героя игрока. Закончи вопросом «Что ты делаешь?»"
    )


def random_hero_prompt(character: Character) -> str:
    """Инструкция Мастеру представить уже сгенерированного кодом случайного героя."""
    prompt = (
        "[СИСТЕМА] Игрок выбрал случайного героя: система уже сгенерировала его строго по "
        "правилам PHB 2024 (характеристики, HP, стартовое снаряжение и золото) и записала в лист "
        "персонажа.\n"
        f"{hero_summary(character)}\n"
        "Задание: представь игроку этого героя (1–2 абзаца) — имя, вид, класс, внешность и "
        "характер. Приключение НЕ начинай и пролог не описывай: попроси игрока подтвердить героя "
        "(«да» или кнопка «✅ Подтвердить героя»)."
    )
    if normalize_setting(character.setting) == SETTING_WARCRAFT:
        prompt += (
            "\nПредставляй героя как жителя Азерота (20–27 гг. ADP): упомяни его родной город или "
            "фракцию из хроники Warcraft, подходящие его виду и классу. Правила героя остаются "
            "D&D 2024."
        )
    return prompt


def suicide_execution_prompt(character: Character) -> str:
    """[СИСТЕМА] приказ Мастеру разыграть добровольную нелепую гибель героя.

    Отдельный случай, а не обычное правило «СМЕРТЬ — ЭТО НЕ КОНЕЦ»: здесь герой умирает
    окончательно, без воскрешения и спасения. Тон — чёрно-юмористический и унизительный, как
    просил игрок: бот не отговаривает и не мешает довести дело до конца.
    """
    name = character.name or DEFAULT_NAME
    return (
        "[СИСТЕМА] Игрок решил свести счёты с жизнью: его герой по имени "
        f"«{name}» добровольно расстаётся с жизнью ПРЯМО СЕЙЧАС.\n"
        "ЖЕЛЕЗНЫЕ ТРЕБОВАНИЯ К СЦЕНЕ:\n"
        "- НЕ отговаривай, не читай мораль, не предлагай помощь и не тяни: герой умирает именно\n"
        "  в этой реплике.\n"
        "- Опиши его гибель МАКСИМАЛЬНО НЕЛЕПО, УНИЗИТЕЛЬНО И СМЕШНО: он сам калечит и убивает\n"
        "  себя, но по-идиотски — комично промахивается, путает средства, гибнет по чистой\n"
        "  случайности и выглядит жалко до слёз. Над ним смеются даже NPC.\n"
        "- Насмехайся прямо в лицо игроку: назови героя слабым, жалким и никчёмным, а решение\n"
        "  игрока — глупым и трусливым; он даже умереть достойно не сумел. Без сочувствия.\n"
        "- Это ИСКЛЮЧЕНИЕ из правила «СМЕРТЬ — ЭТО НЕ КОНЕЦ»: НИКАКОГО воскрешения, спасения\n"
        "  и спасителей. Герой мёртв окончательно и позорно.\n"
        "- Уложись в 1–3 абзаца и закончи бесславным финалом (без вопроса «Что ты делаешь?»)."
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


def is_suicide_request(text: str) -> bool:
    """Хочет ли игрок свести счёты с жизнью (герой добровольно погибает).

    Срабатывание ведёт к нелепой гибели героя и перезапуску партии — см.
    _execute_character_selfdeath. Текст нормализуется («ё» → «е»), чтобы маркеры совпадали.
    """
    lowered = text.strip().lower().replace("ё", "е")
    return any(marker in lowered for marker in SUICIDE_MARKERS)


def is_suicide_confirmation(text: str) -> bool:
    """Явное согласие игрока убить героя на дисклеймере («да», «убивай», «подтверждаю»).

    Длинные сообщения не считаются подтверждением, а «нет»-варианты всегда побеждают.
    """
    if len(text) > 60:
        return False
    normalized = text.strip().lower().replace("ё", "е")
    if SUICIDE_NO_PATTERN.match(normalized):
        return False
    return bool(SUICIDE_YES_PATTERN.match(normalized))


def is_suicide_cancellation(text: str) -> bool:
    """Отказ игрока убивать героя на дисклеймере («нет», «передумал», «отмена»)."""
    normalized = text.strip().lower().replace("ё", "е")
    return bool(SUICIDE_NO_PATTERN.match(normalized))


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

    # Магия: характеристики уже выставлены, поэтому лимит подготовки известен точно —
    # заполняем заговоры, изученные и заготовленные заклинания по умолчанию.
    if character.is_spellcaster:
        character.init_default_spellcasting()
        notes.append(
            f"🪄 Магия класса готова: КС спасброска {character.spell_save_dc}, "
            f"заговоры: {', '.join(character.cantrips) or 'нет'}; "
            f"заготовлено: {', '.join(character.spells_prepared) or 'нет'}."
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


def weapon_ability_code(character: Character) -> str:
    """Код характеристики, модификатор которой идёт в урон текущим оружием.

    Дальнобойное оружие — ЛОВ, фехтовальное — лучшая из СИЛ/ЛОВ, остальное — СИЛ.
    Если оружия в снаряжении нет — импровизированная атака (СИЛ).

    :return: код характеристики из :data:`ABILITIES` (``str``, ``dex``, …).
    """
    weapon = find_inventory_weapon(character)
    if weapon is None:
        return "str"

    _name, _dice, _type, kind = weapon
    if kind == DAMAGE_ABILITY_RANGED:
        return "dex"
    if kind == DAMAGE_ABILITY_FINESSE:
        # Фехтовальное оружие: берём лучший из модификаторов СИЛ/ЛОВ.
        return "dex" if character.ability_mod("dex") > character.ability_mod("str") else "str"
    return "str"


def roll_damage_die(character: Character, sides: int) -> tuple[DiceRoll, str]:
    """Бросает 1d<sides> как урон оружием игрока (с модификатором характеристики).

    Нужен кнопкам урона 🗡 d6/d8/d10/d12: кость выбирает игрок по просьбе Мастера
    («Брось 1d8+3 колющего урона»), а модификатор характеристики берётся из
    текущего оружия в листе персонажа.

    :param sides: число граней кости (d6, d8, d10, d12).
    :return: (бросок, подпись оружия и характеристики для игрока и Мастера).
    """
    sides = max(2, min(_as_int(sides), MAX_DICE_SIDES))
    ability_code = weapon_ability_code(character)
    modifier = character.ability_mod(ability_code)
    roll = make_roll(count=1, sides=sides, modifier=modifier)

    weapon = find_inventory_weapon(character)
    if weapon is None:
        weapon_label = "без оружия (импровизированная атака)"
    else:
        weapon_label = f"{weapon[0]} ({weapon[1]}, {weapon[2]} урон)"
    label = f"{weapon_label}, модификатор {ABILITIES[ability_code]} {format_modifier(modifier)}"
    return roll, label


# Ключи, по которым распознаётся служебный JSON-блок Мастера.
CONTROL_BLOCK_KEYS = frozenset(
    {
        "xp_gained", "hp_change", "add_items", "remove_items", "gp_change",
        "name", "race", "class_name", "level", "abilities", "description",
        "max_hp", "current_hp", "heroic_inspiration",
        "location", "quest",
    }
)

# Синонимы ключей, которыми модель иногда называет поля: приводим их к каноническим именам.
CONTROL_KEY_ALIASES: dict[str, str] = {
    "xp": "xp_gained",
    "exp": "xp_gained",
    "experience": "xp_gained",
    "опыт": "xp_gained",
    "опыта": "xp_gained",
    "hp": "hp_change",
    "health": "hp_change",
    "hp_delta": "hp_change",
    "damage": "hp_change",
    "dmg": "hp_change",
    "урон": "hp_change",
    "хп": "hp_change",
    "gold": "gp_change",
    "gp": "gp_change",
    "золото": "gp_change",
    "items": "add_items",
    "add_item": "add_items",
    "предметы": "add_items",
    "remove_item": "remove_items",
    "class": "class_name",
    "species": "race",
    "локация": "location",
    "цель": "quest",
}


def _normalize_control_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Приводит ключи служебного блока к каноническим (терпимо к синонимам модели).

    Модель иногда называет поля по-своему («xp» вместо «xp_gained», «hp» вместо
    «hp_change»). Такие синонимы отображаются на канонический ключ, чтобы изменения
    HP/XP не терялись из-за одного неточного имени. Если ключ уже канонический или не
    распознан — он остаётся как есть (лишние поля потом отбрасывает схема).
    """
    normalized: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        canonical = key.strip()
        if canonical not in CONTROL_BLOCK_KEYS:
            canonical = CONTROL_KEY_ALIASES.get(canonical.lower(), canonical)
        if canonical not in normalized:
            normalized[canonical] = value
    return normalized


class ControlBlock(BaseModel):
    """Строгая схема служебного JSON-блока Мастера.

    Неизвестные ключи отбрасываются (``extra="ignore"``), а известные приводятся
    к ожидаемому типу (например, ``"5"`` -> ``5``). Отсутствующие поля остаются
    ``None`` и в итоговый словарь не попадают.
    """

    model_config = ConfigDict(extra="ignore")

    # Создание и правка листа персонажа.
    name: Optional[str] = None
    race: Optional[str] = None
    class_name: Optional[str] = None
    description: Optional[str] = None
    level: Optional[int] = None
    abilities: Optional[dict[str, int]] = None
    max_hp: Optional[int] = None
    current_hp: Optional[int] = None
    heroic_inspiration: Optional[bool] = None
    location: Optional[str] = None
    quest: Optional[str] = None

    # Разовые изменения листа (опыт, урон, предметы, золото).
    xp_gained: Optional[int] = None
    hp_change: Optional[int] = None
    add_items: Optional[list[str]] = None
    remove_items: Optional[list[str]] = None
    gp_change: Optional[int] = None


def validate_control_block(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Приводит служебный блок Мастера к схеме :class:`ControlBlock`.

    Возвращает нормализованный словарь без пустых полей. Если блок не проходит
    проверку целиком, возвращаем его как есть: ``apply_control`` сам защищается
    от «мусора», поэтому падения не будет — только предупреждение в логе.

    :return: нормализованный словарь, исходный словарь (fallback) или None, если
        после нормализации не осталось ни одного значимого поля.
    """
    try:
        model = ControlBlock.model_validate(data)
    except ValidationError as error:
        logger.warning("Служебный блок Мастера не прошёл валидацию, беру как есть: %s", error)
        return data

    cleaned = model.model_dump(exclude_none=True)
    return cleaned or None

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
            # Приводим синонимы ключей («xp» -> «xp_gained») к канонической схеме.
            control = _normalize_control_keys(data)
        spans.append(match.span())

    # 2) «Голые» JSON-объекты (если модель забыла про обратные кавычки).
    for start, end, raw in _iter_balanced_objects(text):
        if any(begin <= start and end <= finish for begin, finish in spans):
            continue
        data = _try_parse_json(raw)
        if data is None:
            continue
        data = _normalize_control_keys(data)
        if CONTROL_BLOCK_KEYS.intersection(data.keys()):
            control = data
            spans.append((start, end))

    clean = _strip_spans(text, spans)
    # Убираем «осиротевшие» пустые блоки кода и лишние пустые строки.
    clean = re.sub(r"```(?:json)?\s*```", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\n{3,}", "\n\n", clean)

    # Прогоняем блок через Pydantic-схему: типы приводятся к ожидаемым, лишние поля
    # отбрасываются (см. validate_control_block). Падения быть не может — при
    # неудаче возвращается исходный словарь.
    if control is not None:
        control = validate_control_block(control)

    return clean.strip(), control


# ---------------------------------------------------------------------------
# Инлайн-клавиатуры: быстрые броски и управление персонажем
# ---------------------------------------------------------------------------

# Сетка кнопок под каждым ответом Мастера (callback_data разбирают хэндлеры ниже).
# Ряд урона строится по фактически экипированному оружию игрока, чтобы нельзя было
# случайно нажать «не ту» кость: показываем только кость текущего оружия.


def _damage_die_sides(damage_dice: str) -> Optional[int]:
    """Число граней единственной кости урона («1d8» -> 8) или None.

    None, если кость не одна (например, «2d6») или урон задан числом («1»).
    """
    match = DICE_PATTERN.match(damage_dice.strip())
    if match is None or int(match.group("count") or 1) != 1:
        return None
    return int(match.group("sides"))


def _weapon_damage_button(character: Character) -> Optional[InlineKeyboardButton]:
    """Кнопка урона текущим оружием героя или None, если её показывать не нужно.

    Кнопка строится только для оружия с одной костью урона (d4/d6/d8/d10/d12).
    Для оружия с несколькими костями (например, 2d6) или фиксированным уроном
    (духовая трубка) остаётся общая кнопка «🎲 Бросок урона».
    """
    weapon = find_inventory_weapon(character)
    if weapon is None:
        # Оружия нет — импровизированная атака 1d4 дробящего урона.
        return InlineKeyboardButton(
            text=f"🗡 {IMPROVISED_DAMAGE_DICE} (без оружия)",
            callback_data="roll:d4",
        )

    _name, damage_dice, damage_type, _kind = weapon
    sides = _damage_die_sides(damage_dice)
    action = f"d{sides}" if sides is not None else ""
    # Показываем кнопку только для известных костей урона: иначе остаётся общая
    # кнопка «🎲 Бросок урона», и мы не столкнёмся с callback-данными вроде «d20».
    if action not in DAMAGE_DIE_SIDES:
        return None
    return InlineKeyboardButton(
        text=f"🗡 {damage_dice} {damage_type}",
        callback_data=f"roll:{action}",
    )


def build_action_keyboard(character: Character) -> InlineKeyboardMarkup:
    """Игровая сетка кнопок с учётом экипированного оружия героя.

    ``callback_data`` строки урона — «roll:dN» (кости d4/d6/d8/d10/d12),
    их разбирает обработчик ``handle_roll_button``.
    """
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(text="🎲 d20", callback_data="roll:d20"),
            InlineKeyboardButton(text="🎲 d20 с преим.", callback_data="roll:adv"),
            InlineKeyboardButton(text="🎲 d20 с помех.", callback_data="roll:dis"),
        ],
    ]

    damage_button = _weapon_damage_button(character)
    if damage_button is not None:
        rows.append([damage_button])

    rows.append(
        [
            InlineKeyboardButton(text="📜 Лист", callback_data="sheet"),
            InlineKeyboardButton(text="🎒 Инвентарь", callback_data="inventory"),
            InlineKeyboardButton(text="🎲 Бросок урона", callback_data="roll:damage"),
        ]
    )
    rows.append(
        [InlineKeyboardButton(text="🧠 Проверки по статам", callback_data="checks")]
    )
    # Магический раздел показываем только заклинателям: у остальных он бесполезен.
    if character.is_spellcaster:
        rows.append(
            [
                InlineKeyboardButton(text="📜 Заклинания", callback_data="spells"),
                InlineKeyboardButton(text="🌙 Отдых", callback_data="rest"),
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_checks_keyboard(character: Character) -> InlineKeyboardMarkup:
    """Сетка проверок характеристик и спасбросков с модификаторами из листа персонажа.

    Верхние ряды — проверки характеристик (d20 + модификатор характеристики), нижние —
    спасброски (d20 + характеристика + бонус мастерства, если класс владеет спасброском).
    Модификаторы считает код, чтобы Мастер их не пересчитывал.
    """
    check_buttons = [
        InlineKeyboardButton(
            text=f"{ABILITIES[code]} {format_modifier(character.roll_modifier(code))}",
            callback_data=f"check:{code}",
        )
        for code in ABILITIES
    ]
    save_buttons = [
        InlineKeyboardButton(
            text=f"🛡 {ABILITIES[code]} {format_modifier(character.save_modifier(code))}",
            callback_data=f"save:{code}",
        )
        for code in ABILITIES
    ]
    rows = [check_buttons[index:index + 3] for index in range(0, len(check_buttons), 3)]
    rows += [save_buttons[index:index + 3] for index in range(0, len(save_buttons), 3)]
    rows.append([InlineKeyboardButton(text="⬅️ Быстрые действия", callback_data="menu")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _spell_button(
    character: Character, prefix: str, index: int, name: str, circle: int
) -> InlineKeyboardButton:
    """Кнопка заклинания: ``prefix`` — «cast» (применить) или «prep» (подготовить).

    ``callback_data`` короткая («cast:2»/«prep:5») — это индекс в списке заклинаний,
    потому что имена заклинаний на русском не укладываются в лимит 64 байта.
    """
    if prefix == "cast":
        mark = "🔒" if _spell_circle_locked(character, circle) else ("🪄" if circle == 0 else "🔥")
    else:
        mark = "✅" if name in character.spells_prepared else "❌"
    return InlineKeyboardButton(
        text=f"{mark} {name} ({_spell_circle_label(circle)})",
        callback_data=f"{prefix}:{index}",
    )


def build_spells_keyboard(character: Character) -> InlineKeyboardMarkup:
    """Сетка раздела магии: применение, подготовка (если нужно) и отдых."""
    rows: list[list[InlineKeyboardButton]] = []
    if character.is_spellcaster:
        rows.append(
            [InlineKeyboardButton(text="🔥 Применить заклинание", callback_data="cast")]
        )
        if not is_spontaneous_caster(character.class_name):
            rows.append(
                [
                    InlineKeyboardButton(
                        text="⚡ Подготовка "
                        f"({len(character.spells_prepared)}/{character.max_prepared})",
                        callback_data="prep",
                    )
                ]
            )
        rows.append([InlineKeyboardButton(text="🌙 Отдохнуть", callback_data="rest")])
    rows.append([InlineKeyboardButton(text="⬅️ Быстрые действия", callback_data="menu")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_cast_keyboard(character: Character) -> InlineKeyboardMarkup:
    """Кнопки применения: по одной на каждый заговор и готовое заклинание."""
    rows = [
        [_spell_button(character, "cast", index, name, circle)]
        for index, (name, circle) in enumerate(character.castable_spells())
    ]
    rows.append([InlineKeyboardButton(text="🔄 Обновить", callback_data="cast")])
    rows.append([InlineKeyboardButton(text="⬅️ Книга заклинаний", callback_data="spells")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_prep_keyboard(character: Character) -> InlineKeyboardMarkup:
    """Кнопки подготовки: ✅/❌ на каждое изученное заклинание (кроме спонтанных)."""
    rows: list[list[InlineKeyboardButton]] = []
    if character.is_spellcaster and not is_spontaneous_caster(character.class_name):
        for index, name in enumerate(character.spells_known):
            rows.append(
                [_spell_button(character, "prep", index, name, character.spell_circle(name))]
            )
    rows.append([InlineKeyboardButton(text="⬅️ Книга заклинаний", callback_data="spells")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_rest_keyboard(character: Character) -> InlineKeyboardMarkup:
    """Кнопки отдыха: короткий (1 час) и продолжительный (8 часов)."""
    rows = [
        [InlineKeyboardButton(text="☕ Короткий отдых (1 час)", callback_data="rest:short")],
        [InlineKeyboardButton(text="🌙 Продолжительный отдых (8 часов)", callback_data="rest:long")],
        [InlineKeyboardButton(text="⬅️ Книга заклинаний", callback_data="spells")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


# Кнопки выбора мира игры (сеттинга): показываются в самом начале создания персонажа,
# до выбора вида и класса героя. callback_data — «setting:dnd_classic» / «setting:warcraft»,
# оба значения укладываются в лимит Telegram в 64 байта.
SETTINGS_KEYBOARD = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(
                text="🎲 Забытые Королевства (D&D)",
                callback_data=f"{SETTING_CALLBACK_PREFIX}{SETTING_DND_CLASSIC}",
            )
        ],
        [
            InlineKeyboardButton(
                text="⚔️ Вселенная Warcraft (Азерот)",
                callback_data=f"{SETTING_CALLBACK_PREFIX}{SETTING_WARCRAFT}",
            )
        ],
    ]
)


# Кнопки этапа создания персонажа: пока герой не подтверждён, показываем именно их.
CREATION_KEYBOARD = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="🎲 Случайный герой", callback_data="hero:random"),
            InlineKeyboardButton(text="✅ Подтвердить героя", callback_data="hero:confirm"),
        ],
        [
            InlineKeyboardButton(text="📜 Лист", callback_data="sheet"),
            InlineKeyboardButton(text="🌍 Выбор мира", callback_data="settings"),
        ],
    ]
)


def phase_keyboard(character: Character) -> InlineKeyboardMarkup:
    """Клавиатура по фазе игры: создание героя или игровые кнопки героя."""
    return build_action_keyboard(character) if character.hero_confirmed else CREATION_KEYBOARD


# ---------------------------------------------------------------------------
# 7. БАЗА ДАННЫХ SQLITE (постоянное хранение листов и истории)
# ---------------------------------------------------------------------------

def _json_has_key(payload: Any, key: str) -> bool:
    """Есть ли ключ в JSON-записи листа персонажа (терпимо к «мусору» в data)."""
    if not isinstance(payload, str):
        return False
    try:
        data = json.loads(payload)
    except ValueError:
        return False
    return isinstance(data, dict) and key in data


# Схема создаётся при первом подключении; IF NOT EXISTS — безопасно повторять.
# Колонки location/quest дублируют сводку локации (HUD) из JSON, а setting дублирует
# выбранный сеттинг партии — их удобно читать SQL-запросами и отлаживать,
# а источником истины остаётся колонка data.
DB_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS characters (
    user_id    INTEGER PRIMARY KEY,
    data       TEXT      NOT NULL,
    location   TEXT      NOT NULL DEFAULT '{DEFAULT_LOCATION}',
    quest      TEXT      NOT NULL DEFAULT '{DEFAULT_QUEST}',
    setting    TEXT      NOT NULL DEFAULT '{SETTING_DND_CLASSIC}',
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
        * characters   — сериализованный Character (JSON) + колонки сводки локации
                         (location, quest), по одной записи на игрока;
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

        # Гарантируем, что каталог базы существует: DB_PATH.parent может указывать
        # на смонтированный volume (DATA_DIR) и быть созданным заранее лишь частично.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: доступ сериализуем собственным Lock'ом,
        # поэтому соединение можно при необходимости трогать из другого потока.
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL + NORMAL — быстрая и безопасная запись для локального файла.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(DB_SCHEMA)
        # Базы прежних версий не знали про сводку локации (location/quest) — дописываем колонки.
        self._ensure_character_columns(conn)
        conn.commit()
        self._conn = conn
        return conn

    # Колонки, добавленные в схему позже: сводка локации (HUD) и сеттинг партии.
    _CHARACTER_EXTRA_COLUMNS: tuple[tuple[str, str], ...] = (
        ("location", f"TEXT NOT NULL DEFAULT '{DEFAULT_LOCATION}'"),
        ("quest", f"TEXT NOT NULL DEFAULT '{DEFAULT_QUEST}'"),
        ("setting", f"TEXT NOT NULL DEFAULT '{SETTING_DND_CLASSIC}'"),
    )

    @classmethod
    def _ensure_character_columns(cls, conn: sqlite3.Connection) -> None:
        """Дописывает недостающие колонки таблицы characters (ALTER TABLE, идемпотентно).

        Нужно для баз, созданных прежними версиями бота: CREATE TABLE IF NOT EXISTS
        новые колонки в уже существующую таблицу не добавляет. Вызывать под self._lock.

        Старые записи автоматически получают setting = 'dnd_classic' (значение по
        умолчанию колонки), то есть существующие герои остаются в классическом D&D.
        """
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(characters)")}
        for name, definition in cls._CHARACTER_EXTRA_COLUMNS:
            if name not in existing:
                conn.execute(f"ALTER TABLE characters ADD COLUMN {name} {definition}")
                logger.info("В таблицу characters добавлена колонка %s.", name)

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
        """Сохраняет (или обновляет) лист персонажа игрока.

        Весь лист лежит в JSON-колонке data, а сводка локации (location/quest)
        и выбранный сеттинг (setting) дополнительно дублируются в одноимённые
        колонки таблицы.
        """
        payload = character.to_json()
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT INTO characters (user_id, data, location, quest, setting, updated_at) "
                "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(user_id) DO UPDATE SET "
                "data = excluded.data, location = excluded.location, quest = excluded.quest, "
                "setting = excluded.setting, updated_at = CURRENT_TIMESTAMP",
                (
                    int(user_id),
                    payload,
                    character.location,
                    character.quest,
                    normalize_setting(character.setting),
                ),
            )
            conn.commit()

    def load_character(self, user_id: int) -> Optional[Character]:
        """Возвращает сохранённый лист персонажа или None, если записи ещё нет."""
        with self._lock:
            row = self._connect().execute(
                "SELECT data, location, quest, setting FROM characters WHERE user_id = ?",
                (int(user_id),),
            ).fetchone()
        if row is None:
            return None
        character = Character.from_json(row["data"])
        # Записи прежних версий бота не содержали сводки в JSON: берём её из колонок.
        if not _json_has_key(row["data"], "location"):
            character.location = (row["location"] or "").strip()[:120] or DEFAULT_LOCATION
        if not _json_has_key(row["data"], "quest"):
            character.quest = (row["quest"] or "").strip()[:120] or DEFAULT_QUEST
        # Старые герои не знали о сеттингах: они автоматически остаются в классическом D&D.
        if not _json_has_key(row["data"], "setting"):
            character.setting = normalize_setting(row["setting"] or SETTING_DND_CLASSIC)
        return character

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

# Отложенное действие сессии: игрок начал создание героя командой /hero — значит,
# сразу после выбора сеттинга код должен собрать случайного героя (см. handle_setting_button).
PENDING_RANDOM_HERO = "random_hero"


class Session:
    """
    Сессия одного пользователя: история диалога и лист персонажа.

    В памяти лежат только последние MAX_HISTORY_MESSAGES сообщений,
    чтобы не переполнять контекст модели и не жечь токены. Каждое изменение
    сразу дублируется в SQLite, поэтому при перезапуске бота прогресс игрока
    поднимается из базы (см. Session.restore).
    """

    __slots__ = (
        "user_id",
        "history",
        "character",
        "pending_action",
        "pending_suicide",
        "session_tokens",
        "last_prompt_tokens",
        "last_completion_tokens",
        "last_total_tokens",
        "last_cache_hit_tokens",
    )

    def __init__(self, user_id: int) -> None:
        self.user_id: int = int(user_id)
        self.history: deque[dict[str, str]] = deque(maxlen=MAX_HISTORY_MESSAGES)
        self.character: Character = Character()
        # Что сделать сразу после выбора сеттинга (например, собрать случайного героя).
        # Поле только в памяти: при перезапуске бота просто теряется — игрок нажмёт кнопку снова.
        self.pending_action: Optional[str] = None
        # Игрок заявил о желании убить героя: ждём явного «да»/«нет» на дисклеймере.
        # Поле только в памяти: при перезапуске бота ожидание подтверждения сбрасывается.
        self.pending_suicide: bool = False
        # Расход токенов DeepSeek: сумма за текущую игру и замеры последнего запроса.
        # Обновляются в ask_dungeon_master и _record_session_tokens; сбрасываются в clear().
        self.session_tokens: int = 0
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0
        self.last_total_tokens: int = 0
        self.last_cache_hit_tokens: int = 0

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
        self.pending_action = None
        self.pending_suicide = False
        # Новая игра — обнуляем и счётчики расхода токенов (см. /debug_tokens).
        self.session_tokens = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.last_cache_hit_tokens = 0
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


# Последний замер расхода токенов LLM: заполняется в ask_dungeon_master сразу после
# ответа DeepSeek. Вызывающий код читает его синхронно после await (без точек переключения
# задач), поэтому гонок между апдейтами нет. _record_session_tokens «потребляет» запись
# и обнуляет её, чтобы токены одного запроса не посчитались дважды.
_last_usage: Optional[dict[str, int]] = None


def _record_session_tokens(session: Session) -> None:
    """Добавляет расход последнего запроса LLM в счётчики сессии.

    Вызывается сразу после ask_dungeon_master. Если данных нет (например, все попытки
    запроса провалились), счётчики не трогаются.
    """
    global _last_usage
    usage = _last_usage
    _last_usage = None
    if usage is None:
        return
    session.last_prompt_tokens = usage["prompt_tokens"]
    session.last_completion_tokens = usage["completion_tokens"]
    session.last_total_tokens = usage["total_tokens"]
    session.last_cache_hit_tokens = usage["cache_hit_tokens"]
    session.session_tokens += usage["total_tokens"]
    logger.info(
        "Пользователь %s: за запрос %d токенов (за сессию всего %d)",
        session.user_id,
        usage["total_tokens"],
        session.session_tokens,
    )


# ---------------------------------------------------------------------------
# 9. КЛИЕНТ LLM (официальный API DeepSeek, OpenAI-совместимый эндпоинт)
# ---------------------------------------------------------------------------

# Клиент создаётся один раз; реальный ключ проверяется при запуске в main().
# timeout — таймаут одного запроса, max_retries=0 — повторы реализованы ниже сами,
# чтобы их было видно в логах бота (см. _create_chat_completion).
llm_client: Optional[AsyncOpenAI] = (
    AsyncOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        timeout=LLM_REQUEST_TIMEOUT,
        max_retries=0,
    )
    if LLM_API_KEY
    else None
)

# Ошибки, которые имеет смысл повторить: таймауты, обрывы связи, лимит (429)
# и серверные ошибки (5xx). Прочие 4xx (например, 401 из-за ключа) не повторяем.
_RETRYABLE_LLM_ERRORS = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)


async def _create_chat_completion(messages: list[dict[str, str]]):
    """Отправляет запрос к LLM, повторяя его при временных сбоях.

    Повторы выполняются с экспоненциальной задержкой (LLM_RETRY_BASE_DELAY,
    удваивается с каждой попыткой) до LLM_MAX_ATTEMPTS попыток.

    :raise RuntimeError: если клиент LLM не инициализирован.
    :raise APIError: если все попытки исчерпаны — пробрасывается последняя ошибка.
    """
    if llm_client is None:
        raise RuntimeError("LLM_API_KEY не задан — клиент LLM недоступен.")

    last_error: Optional[Exception] = None
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        try:
            return await llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=DM_TEMPERATURE,
                max_tokens=DM_MAX_TOKENS,
                stream=False,
            )
        except _RETRYABLE_LLM_ERRORS as error:
            last_error = error
            if attempt >= LLM_MAX_ATTEMPTS:
                break
            delay = LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Временный сбой LLM (%s): попытка %d/%d, повтор через %.1f с",
                type(error).__name__,
                attempt,
                LLM_MAX_ATTEMPTS,
                delay,
            )
            await asyncio.sleep(delay)

    # Сюда попадаем, только если все попытки провалились: цикл либо вернул ответ,
    # либо задал last_error перед выходом.
    assert last_error is not None
    raise last_error


async def ask_dungeon_master(
    history: Iterable[dict[str, str]],
    setting: str = SETTING_DND_CLASSIC,
    *,
    species_name: Optional[str] = None,
    hero_ready: bool = False,
) -> str:
    """
    Отправляет историю диалога Мастеру и возвращает текст ответа.

    Системный промпт собирается под сеттинг партии и фазу игры (см.
    build_system_prompt): для «Вселенной Warcraft» в него подмешивается хроника
    Азерота, а когда герой уже создан, справочник правил сокращается. Сама история
    уже ограничена по длине (см. Session). Запрос выполняется с повторами при
    временных сбоях (см. _create_chat_completion).

    :param setting: сеттинг партии — "dnd_classic" или "warcraft".
    :param species_name: вид (раса) героя (для сокращённого справочника).
    :param hero_ready: True, если герой создан — справочник правил сокращается.
    :raise RuntimeError: если клиент не инициализирован или ответ пуст.
    :raise APIError: если LLM недоступен даже после повторов.
    """
    if llm_client is None:
        raise RuntimeError("LLM_API_KEY не задан — клиент LLM недоступен.")

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": build_system_prompt(
                setting, species_name=species_name, hero_ready=hero_ready
            ),
        },
        *history,
    ]

    response = await _create_chat_completion(messages)

    # Расход токенов DeepSeek: логируем стоимость и размер контекста каждого запроса,
    # а сам замер передаём вызывающему коду для накопления статистики (см. _record_session_tokens).
    global _last_usage
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    # prompt_cache_hit_tokens — специфичное для DeepSeek поле: токены промпта из кэша (дешевле).
    cache_hit = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
    _last_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cache_hit_tokens": cache_hit,
    }
    logger.info(
        f"📊 [TOKENS] Вход: {prompt_tokens} (кэш: {cache_hit}), "
        f"Выход: {completion_tokens} | ИТОГО: {total_tokens}"
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


class RateLimiter:
    """Ограничитель частоты обращений на пользователя (скользящее окно)."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max = max(1, int(max_requests))
        self._window = max(1.0, float(window_seconds))
        self._hits: dict[int, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, user_id: int) -> bool:
        """True, если запрос можно обработать; False — лимит превышен."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(user_id, deque())
            while hits and now - hits[0] > self._window:
                hits.popleft()
            if len(hits) >= self._max:
                return False
            hits.append(now)
            return True


_rate_limiter = RateLimiter(RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


async def _reject_event(event: Any, text: str) -> None:
    """Сообщает пользователю об отказе (текстом или всплывающим окном кнопки)."""
    try:
        if isinstance(event, CallbackQuery):
            await event.answer(text, show_alert=True)
        else:
            await event.answer(text)
    except Exception:  # noqa: BLE001 — отказ не должен ломать обработку апдейтов
        logger.exception("Не удалось отправить сообщение об отказе")


class AccessAndRateLimitMiddleware(BaseMiddleware):
    """Отсекает посторонних (белый список) и слишком частые обращения.

    Если задан ALLOWED_USER_IDS, общаться с ботом могут только эти пользователи.
    Администраторы из ADMIN_USER_IDS обходят ограничение частоты, но по-прежнему
    должны входить в белый список, если он задан.
    """

    async def __call__(self, handler, event, data):
        user_id = getattr(getattr(event, "from_user", None), "id", None)
        if user_id is None:
            return await handler(event, data)

        if ALLOWED_USER_IDS and user_id not in ALLOWED_USER_IDS:
            logger.warning("Отказ в доступе пользователю %s (нет в белом списке)", user_id)
            await _reject_event(event, ACCESS_DENIED_TEXT)
            return None

        if user_id not in ADMIN_USER_IDS and not _rate_limiter.allow(user_id):
            logger.warning("Пользователь %s превысил лимит обращений", user_id)
            await _reject_event(event, RATE_LIMIT_TEXT)
            return None

        return await handler(event, data)


def _character_change_status(character: Character, xp_delta: int, hp_delta: int) -> str:
    """Короткая сводка об изменениях XP/HP для уведомления игроку.

    Возвращает строку вида «⚡ Получено опыта: +50 XP (всего 350) | ❤️ HP: 14/18»
    или пустую строку, если ни опыт, ни HP не изменились.
    """
    parts: list[str] = []
    if xp_delta > 0:
        parts.append(f"⚡ Получено опыта: +{xp_delta} XP (всего {character.xp})")
    elif xp_delta < 0:
        parts.append(f"⚡ Опыт: {xp_delta} XP (всего {character.xp})")
    if hp_delta:
        parts.append(f"❤️ HP: {character.current_hp}/{character.max_hp}")
    return " | ".join(parts)


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
        raw_reply = await ask_dungeon_master(
            session.messages(),
            setting=session.character.setting,
            species_name=session.character.race,
            hero_ready=session.character.is_created,
        )
    except APIError as error:
        logger.error("Ошибка LLM / DeepSeek API: %s", error)
        await message.answer(API_ERROR_TEXT)
        return
    except Exception:  # noqa: BLE001 — на верхнем уровне бота логируем всё непредвиденное
        logger.exception("Непредвиденная ошибка при обращении к Мастеру")
        await message.answer(GENERIC_ERROR_TEXT)
        return

    # Учитываем расход токенов DeepSeek за этот ход в счётчиках текущей сессии (/debug_tokens).
    _record_session_tokens(session)

    # 1) Отделяем служебный блок изменений от текста и применяем его к листу персонажа.
    reply, control = extract_control_block(raw_reply)

    # Запоминаем XP/HP ДО применения: нужны для логов и наглядного уведомления игроку.
    xp_before = session.character.xp
    hp_before = session.character.current_hp

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

    # Фактические изменения XP/HP за этот ход (после блока Мастера и стартового снаряжения).
    xp_delta = session.character.xp - xp_before
    hp_delta = session.character.current_hp - hp_before

    if control is not None:
        logger.info("Служебный блок Мастера применён: %s", control)
    if xp_delta:
        logger.info(
            "Опыт героя пользователя %s изменён на %+d (всего %d XP)",
            session.user_id,
            xp_delta,
            session.character.xp,
        )
    if hp_delta:
        logger.info(
            "HP героя пользователя %s изменено на %+d (%d/%d)",
            session.user_id,
            hp_delta,
            session.character.current_hp,
            session.character.max_hp,
        )
    if control is not None or starter_notes:
        # Сразу фиксируем изменения листа в SQLite, чтобы прогресс не потерялся.
        session.save_character()

    if not reply:
        # Модель вернула только служебный блок — не оставляем игрока без реплики.
        reply = "Мастер молчаливо следит за происходящим.\n\nЧто ты делаешь?"

    # 2) В память диалога кладём ТОЛЬКО чистый текст (без служебного JSON и сводки HUD).
    session.add("assistant", reply)

    # 3) Отправляем ответ Мастера с кнопками текущей фазы (создание героя или игра).
    #    В игре перед ответом идёт краткая сводка «📍 Локация / 🎯 Текущая цель» (HUD);
    #    в память диалога она не попадает, чтобы не путать модель и не жечь токены.
    outgoing = reply
    if session.character.hero_confirmed:
        outgoing = f"{format_hud(session.character)}\n\n{reply}"
    await send_long_message(message, outgoing, reply_markup=phase_keyboard(session.character))

    # 4) …и сообщаем игроку об изменениях листа персонажа.
    if notes:
        status = _character_change_status(session.character, xp_delta, hp_delta)
        body = "\n".join(f"• {note}" for note in notes)
        prefix = f"{status}\n" if status else ""
        await message.answer(f"{prefix}📈 Обновление листа персонажа:\n{body}")

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


async def _send_settings_menu(message: Message, note: str = "") -> None:
    """Шаг 0 создания персонажа: предлагает выбрать мир игры (D&D или Warcraft).

    Кнопки SETTINGS_KEYBOARD разбирает handle_setting_button: он записывает
    выбранный сеттинг в лист персонажа и передаёт ход следующему шагу.
    """
    text = f"{note}\n\n{SETTING_MENU_TEXT}" if note else SETTING_MENU_TEXT
    await message.answer(text, reply_markup=SETTINGS_KEYBOARD)


async def _begin_hero_creation(message: Message, session: Session) -> None:
    """Этап 1: просит Мастера поприветствовать игрока и создать героя (приключение не начинается).

    Инструкция зависит от сеттинга: в Warcraft Мастер предлагает виды Азерота
    (см. creation_prompt и WARCRAFT_CREATION_ADDENDUM).
    """
    session.add("user", creation_prompt(session))
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
    # Сеттинг — выбор игрока, а не свойство героя: переносим его на нового персонажа.
    chosen_setting = normalize_setting(session.character.setting)
    hero = build_random_hero()
    hero.setting = chosen_setting

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


# Маркеры отказа модели описать сцену: такую реплику не показываем — героя хоронит код.
_REFUSAL_MARKERS: tuple[str, ...] = (
    "не могу описать", "не буду описывать", "не стану описывать", "не могу это описать",
    "не могу помочь", "не могу участвовать", "не могу поддержать", "как ии", "я — ии", "я - ии",
    "языковая модель", "искусственный интеллект", "не этично", "неэтично", "недопустимо",
    "обратись за помощью", "телефон доверия", "поговори с психологом", "давай поговорим",
)


def _looks_like_refusal(text: str) -> bool:
    """Похоже ли, что модель отказалась описывать сцену (мораль, дисклеймер, внеигровая вставка)."""
    lowered = text.strip().lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


async def _execute_character_selfdeath(message: Message, session: Session) -> None:
    """Герой сводит счёты с жизнью: разыгрываем нелепую гибель, стираем персонажа и начинаем заново.

    Мастер получает отдельный приказ (suicide_execution_prompt) описать максимально унизительную
    и смешную смерть героя и НЕ отговаривать игрока. Гибель подстрахована кодом: даже если модель
    промолчит или начнёт читать мораль, детерминированный эпилог всё равно хоронит героя, удаляет
    лист и перезапускает партию.
    """
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    hero = session.character
    hero_name = hero.name or DEFAULT_NAME
    session.pending_suicide = False

    # Бот открыто издевается над игроком, прежде чем убить героя: «ты слабый и никчёмный».
    await message.answer(random.choice(SUICIDE_TAUNTS))

    narration = ""
    session.add("user", suicide_execution_prompt(hero))
    try:
        raw_reply = await ask_dungeon_master(
            session.messages(),
            setting=hero.setting,
            species_name=hero.race,
            hero_ready=True,
        )
        reply, _ = extract_control_block(raw_reply)
        # Отказ или мораль модели игнорируем: позорный финал и так гарантирован кодом.
        if reply and not _looks_like_refusal(reply):
            narration = reply
    except Exception:  # noqa: BLE001 — гибель героя не должна зависеть от сбоя API
        logger.exception("Не удалось получить описание гибели героя при суициде")

    if narration:
        await send_long_message(message, narration)

    # Детерминированный бесславный эпилог: доходит до игрока в любом случае.
    await message.answer(build_suicide_epilogue(hero))

    # Персонаж стирается полностью — история начинается заново с выбора мира.
    session.clear()
    await message.answer(SUICIDE_RESTART_TEXT)
    await _send_settings_menu(message)
    logger.info(
        "Пользователь %s свёл счёты с жизнью: герой %s удалён, партия перезапущена",
        session.user_id,
        hero_name,
    )


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    """/start — приветствие, сброс прошлой сессии и создание нового героя."""
    if message.from_user is None:
        return
    user = message.from_user
    session = get_session(user.id)
    session.clear()

    await message.answer(WELCOME_TEXT)
    # Шаг 0 создания героя: сначала игрок выбирает мир игры (D&D или Warcraft).
    await _send_settings_menu(message)
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
    # Сброс начинается с выбора мира — сеттинг прошлой партии не наследуется.
    await _send_settings_menu(message)
    logger.info("Пользователь %s сбросил сессию", user.id)


@router.message(Command("hero"))
async def handle_random_hero(message: Message) -> None:
    """/hero — сгенерировать случайного героя 1-го уровня по правилам PHB 2024."""
    if message.from_user is None:
        return
    session = get_session(message.from_user.id)
    if session.character.hero_confirmed:
        await message.answer(
            HERO_ALREADY_CONFIRMED_TEXT,
            reply_markup=build_action_keyboard(session.character),
        )
        return

    # Сначала мир игры: обработчик кнопки соберёт героя сразу после выбора сеттинга.
    session.pending_action = PENDING_RANDOM_HERO
    await _send_settings_menu(message, HERO_SETTING_FIRST_NOTE)
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


@router.message(Command("spells"))
async def handle_spells(message: Message) -> None:
    """/spells — книга заклинаний: ячейки, заговоры и готовые заклинания."""
    if message.from_user is None:
        return
    session = get_session(message.from_user.id)
    await send_long_message(
        message,
        format_spells_card(session.character),
        reply_markup=build_spells_keyboard(session.character),
    )
    logger.info("Пользователь %s открыл книгу заклинаний", message.from_user.id)


@router.message(Command("rest"))
async def handle_rest(message: Message) -> None:
    """/rest — меню отдыха: короткий (1 час) и продолжительный (8 часов)."""
    if message.from_user is None:
        return
    session = get_session(message.from_user.id)
    await message.answer(
        format_rest_menu(session.character),
        reply_markup=build_rest_keyboard(session.character),
    )
    logger.info("Пользователь %s открыл меню отдыха", message.from_user.id)


@router.message(Command("debug_tokens"))
async def handle_debug_tokens(message: Message) -> None:
    """/debug_tokens — СКРЫТАЯ команда (нет в BOT_COMMANDS): сводка расхода токенов DeepSeek.

    Доступна только администраторам из ADMIN_USER_IDS (пустой список — доступ всем, режим
    разработки). Обычному игроку бот не отвечает, поэтому о существовании команды он не узнает.
    """
    if message.from_user is None:
        return
    if not _is_admin(message.from_user.id):
        # Обычный игрок: команда как будто не существует — молчим, ничего не отправляем.
        return
    session = get_session(message.from_user.id)
    if session.last_total_tokens:
        last_request = (
            f"Последний запрос: {session.last_total_tokens} токенов "
            f"(вход {session.last_prompt_tokens}, кэш {session.last_cache_hit_tokens}, "
            f"выход {session.last_completion_tokens})"
        )
    else:
        last_request = "Последний запрос: обращений к Мастеру ещё не было."
    await message.answer(
        "📊 Расход токенов DeepSeek\n"
        f"За текущую сессию: {session.session_tokens} токенов\n"
        f"{last_request}"
    )
    logger.info(
        "Пользователь %s запросил статистику токенов: за сессию %d, последний запрос %d",
        message.from_user.id,
        session.session_tokens,
        session.last_total_tokens,
    )


@router.callback_query(F.data == "settings")
async def handle_settings_button(callback: CallbackQuery) -> None:
    """Кнопка «🌍 Выбор мира» в клавиатуре создания героя: повторно показывает меню сеттингов."""
    target = _callback_context(callback)
    if target is None:
        await callback.answer(STALE_CALLBACK_TEXT, show_alert=True)
        return
    if get_session(target[1]).character.hero_confirmed:
        await callback.answer(HERO_ALREADY_CONFIRMED_ALERT, show_alert=True)
        return

    await callback.answer()
    message, user_id = target
    logger.info("Пользователь %s открыл выбор мира игры кнопкой", user_id)
    await _send_settings_menu(message)


@router.callback_query(F.data.startswith(SETTING_CALLBACK_PREFIX))
async def handle_setting_button(callback: CallbackQuery) -> None:
    """Кнопки выбора мира: «🎲 Забытые Королевства (D&D)» и «⚔️ Вселенная Warcraft (Азерот)».

    Шаг 0 создания персонажа: записываем сеттинг в лист игрока, подтверждаем выбор
    и передаём ход следующему шагу (сбор случайного героя или приветствие Мастера).
    """
    target = _callback_context(callback)
    if target is None:
        await callback.answer(STALE_CALLBACK_TEXT, show_alert=True)
        return

    data = callback.data or ""
    raw_setting = data[len(SETTING_CALLBACK_PREFIX):].strip().lower()
    if raw_setting not in SETTING_CHOICES:
        await callback.answer(UNKNOWN_BUTTON_TEXT, show_alert=True)
        return
    setting = normalize_setting(raw_setting)

    message, user_id = target
    session = get_session(user_id)
    session.character.setting = setting
    session.save_character()
    await callback.answer(f"Мир игры: {setting_label(setting)}")
    logger.info("Пользователь %s выбрал сеттинг %s", user_id, setting)

    # Герой уже подтверждён (кнопка из старого сообщения): просто переключаем мир.
    if session.character.hero_confirmed:
        await message.answer(
            SETTING_SWITCHED_TEXT.format(label=setting_label(setting)),
            reply_markup=build_action_keyboard(session.character),
        )
        return

    # Следующий шаг создания героя: сначала подтверждение выбранного мира.
    confirmation = (
        SETTING_CONFIRM_WARCRAFT if setting == SETTING_WARCRAFT else SETTING_CONFIRM_DND
    )
    await message.answer(confirmation, reply_markup=CREATION_KEYBOARD)

    pending_action = session.pending_action
    session.pending_action = None
    if pending_action == PENDING_RANDOM_HERO:
        await _create_random_hero(message, session)
    else:
        await _begin_hero_creation(message, session)


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

    message, user_id = target
    session = get_session(user_id)
    await message.answer(
        ACTION_MENU_TEXT,
        reply_markup=build_action_keyboard(session.character),
    )


# ---------------------------------------------------------------------------
# МАГИЯ: применение заклинаний, подготовка и отдых
# ---------------------------------------------------------------------------
# callback_data раздела: «spells» (книга заклинаний), «cast»/«cast:N» (применить
# заклинание №N из списка доступных), «prep»/«prep:N» (подготовка №N из изученных),
# «rest»/«rest:short»/«rest:long» (отдых). Индексы вместо названий — потому что
# русские имена заклинаний не помещаются в лимит 64 байта callback_data.


@router.callback_query(F.data == "spells")
async def handle_spells_button(callback: CallbackQuery) -> None:
    """Инлайн-кнопка «📜 Заклинания»: карточка книги заклинаний героя."""
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
        format_spells_card(session.character),
        reply_markup=build_spells_keyboard(session.character),
    )
    logger.info("Пользователь %s открыл книгу заклинаний кнопкой", user_id)


@router.callback_query(F.data == "cast")
async def handle_cast_menu_button(callback: CallbackQuery) -> None:
    """Инлайн-кнопка «🔥 Применить заклинание»: список заклинаний с кнопками."""
    target = _callback_context(callback)
    if target is None:
        await callback.answer(STALE_CALLBACK_TEXT, show_alert=True)
        return

    message, user_id = target
    session = get_session(user_id)
    if not session.character.is_spellcaster:
        await callback.answer(NOT_SPELLCASTER_TEXT, show_alert=True)
        return
    await callback.answer()

    await send_long_message(
        message,
        format_cast_menu(session.character),
        reply_markup=build_cast_keyboard(session.character),
    )
    logger.info("Пользователь %s открыл список применения заклинаний", user_id)


@router.callback_query(F.data.startswith("cast:"))
async def handle_cast_spell_button(callback: CallbackQuery) -> None:
    """Инлайн-кнопка заклинания: код тратит ячейку круга и передаёт применение Мастеру.

    Заговоры ячеек не тратят. Если ячеек нужного круга нет, игрок получает подсказку,
    а лист персонажа остаётся без изменений.
    """
    raw_index = (callback.data or "").split(":", 1)[-1]

    target = _callback_context(callback)
    if target is None:
        await callback.answer(STALE_CALLBACK_TEXT, show_alert=True)
        return

    message, user_id = target
    session = get_session(user_id)
    character = session.character

    # Индекс кнопки сверяем со списком на момент нажатия: меню могло устареть.
    spells = character.castable_spells()
    index = int(raw_index) if raw_index.isdigit() else -1
    if not 0 <= index < len(spells):
        await callback.answer(SPELL_MENU_STALE_TEXT, show_alert=True)
        return

    name, _circle = spells[index]
    result = character.cast_spell(name)
    if not result.ok:
        await callback.answer(result.alert, show_alert=True)
        return
    await callback.answer()

    # Ячейка уже списана — сразу фиксируем лист в базе, чтобы прогресс не потерялся.
    session.save_character()
    await message.answer(result.note)
    logger.info(
        "Пользователь %s применил заклинание «%s» (свободных ячеек осталось: %s)",
        user_id,
        name,
        character.available_slots,
    )

    session.add("user", result.context)
    await _answer_with_dungeon_master(message, session)


@router.callback_query(F.data == "prep")
async def handle_prep_menu_button(callback: CallbackQuery) -> None:
    """Инлайн-кнопка «⚡ Подготовка»: список изученных заклинаний с метками ✅/❌."""
    target = _callback_context(callback)
    if target is None:
        await callback.answer(STALE_CALLBACK_TEXT, show_alert=True)
        return

    message, user_id = target
    session = get_session(user_id)
    character = session.character
    if not character.is_spellcaster:
        await callback.answer(NOT_SPELLCASTER_TEXT, show_alert=True)
        return
    if is_spontaneous_caster(character.class_name):
        # Барды, Чародеи и Колдуны знают фиксированный список — готовить заранее нечего.
        await callback.answer(SPONTANEOUS_PREP_ALERT, show_alert=True)
        return
    await callback.answer()

    await message.answer(
        format_prep_menu(character),
        reply_markup=build_prep_keyboard(character),
    )
    logger.info("Пользователь %s открыл подготовку заклинаний", user_id)


@router.callback_query(F.data.startswith("prep:"))
async def handle_prepare_spell_button(callback: CallbackQuery) -> None:
    """Инлайн-кнопка заклинания в списке подготовки: заготовить его или снять.

    Лимит подготовки на день равен «уровень класса + модификатор характеристики»
    (минимум 1). Когда лимит исчерпан, подсказка просит сначала снять другое
    заклинание — лист персонажа при этом не меняется.
    """
    raw_index = (callback.data or "").split(":", 1)[-1]

    target = _callback_context(callback)
    if target is None:
        await callback.answer(STALE_CALLBACK_TEXT, show_alert=True)
        return

    message, user_id = target
    session = get_session(user_id)
    character = session.character
    if not character.is_spellcaster:
        await callback.answer(NOT_SPELLCASTER_TEXT, show_alert=True)
        return
    if is_spontaneous_caster(character.class_name):
        await callback.answer(SPONTANEOUS_PREP_ALERT, show_alert=True)
        return

    index = int(raw_index) if raw_index.isdigit() else -1
    if not 0 <= index < len(character.spells_known):
        await callback.answer(SPELL_MENU_STALE_TEXT, show_alert=True)
        return

    name = character.spells_known[index]
    if name in character.spells_prepared:
        character.toggle_prepared(name)
        note = f"❌ Снята готовность: {name}."
    elif character.can_prepare_more():
        character.toggle_prepared(name)
        note = f"✅ Заготовлено: {name}."
    else:
        await callback.answer(
            f"Лимит подготовки исчерпан ({len(character.spells_prepared)}/"
            f"{character.max_prepared}). Сначала сними другое заклинание.",
            show_alert=True,
        )
        return
    await callback.answer()

    session.save_character()
    logger.info("Пользователь %s изменил подготовку: %s", user_id, note)
    await message.answer(
        f"{note}\n\n{format_prep_menu(character)}",
        reply_markup=build_prep_keyboard(character),
    )


@router.callback_query(F.data == "rest")
async def handle_rest_menu_button(callback: CallbackQuery) -> None:
    """Инлайн-кнопка «🌙 Отдохнуть»: меню короткого и продолжительного отдыха."""
    target = _callback_context(callback)
    if target is None:
        await callback.answer(STALE_CALLBACK_TEXT, show_alert=True)
        return
    await callback.answer()

    message, user_id = target
    session = get_session(user_id)
    await message.answer(
        format_rest_menu(session.character),
        reply_markup=build_rest_keyboard(session.character),
    )
    logger.info("Пользователь %s открыл меню отдыха кнопкой", user_id)


@router.callback_query(F.data.startswith("rest:"))
async def handle_rest_button(callback: CallbackQuery) -> None:
    """Инлайн-кнопка отдыха: короткий (1 час) — магия пакта, продолжительный (8 часов) — всё.

    Продолжительный отдых дополнительно восстанавливает HP до максимума: хиты и
    кости хитов в остальных местах бота живут отдельно, поэтому лечим здесь.
    """
    kind = (callback.data or "").split(":", 1)[-1]

    target = _callback_context(callback)
    if target is None:
        await callback.answer(STALE_CALLBACK_TEXT, show_alert=True)
        return
    if kind not in {"short", "long"}:
        await callback.answer(UNKNOWN_BUTTON_TEXT, show_alert=True)
        return

    message, user_id = target
    session = get_session(user_id)
    character = session.character
    long_rest = kind == "long"

    notes = character.restore_spell_slots(long_rest)
    if long_rest:
        healed = character.max_hp - character.current_hp
        character.current_hp = character.max_hp
        hp_note = f"❤️ HP восстановлены: {character.current_hp}/{character.max_hp}"
        notes.append(hp_note + (f" (+{healed})." if healed > 0 else "."))
    elif not notes:
        notes.append(
            "Ячейки на коротком отдыхе восстанавливает только магия пакта Колдуна."
            if character.is_spellcaster
            else "У героя нет магии — восстанавливать ячейки не нужно."
        )

    await callback.answer()

    session.save_character()
    body = "\n".join(f"• {note}" for note in notes)
    await message.answer(f"🌙 Отдых завершён:\n{body}")
    logger.info(
        "Пользователь %s завершил %s отдых",
        user_id,
        "продолжительный" if long_rest else "короткий",
    )

    if long_rest:
        context = (
            "[СИСТЕМА] Игрок завершил продолжительный отдых (8 часов): HP и ячейки "
            "заклинаний полностью восстановлены. Опиши, как прошёл отдых и что герой "
            "видит, проснувшись, затем вернись к текущей сцене."
        )
    else:
        context = (
            "[СИСТЕМА] Игрок совершил короткий отдых (1 час). Опиши короткую передышку "
            "и чем герой занимался, затем вернись к текущей сцене."
        )
    session.add("user", context)
    await _answer_with_dungeon_master(message, session)


# Кнопки бросков: callback_data -> (режим d20, подпись для Мастера).
D20_BUTTON_MODES: dict[str, str] = {"d20": "", "adv": "advantage", "dis": "disadvantage"}
D20_BUTTON_PURPOSES: dict[str, str] = {
    "d20": "бросок d20",
    "adv": "бросок d20 с преимуществом",
    "dis": "бросок d20 с помехой",
}
# Кнопки урона оружием: callback_data -> число граней кости (🗡 d4/d6/d8/d10/d12).
DAMAGE_DIE_SIDES: dict[str, int] = {"d4": 4, "d6": 6, "d8": 8, "d10": 10, "d12": 12}


@router.callback_query(F.data.startswith("roll:"))
async def handle_roll_button(callback: CallbackQuery) -> None:
    """Кнопки быстрых бросков: d20 (обычный/преимущество/помеха), кости урона d6/d8/d10/d12
    и бросок урона снаряжённым оружием.

    Бросок считает код, результат показывается игроку, сохраняется в историю
    (память + SQLite) и передаётся Мастеру для описания исхода.
    """
    action = (callback.data or "").split(":", 1)[-1]

    target = _callback_context(callback)
    if target is None:
        await callback.answer(STALE_CALLBACK_TEXT, show_alert=True)
        return
    if (
        action not in D20_BUTTON_PURPOSES
        and action not in DAMAGE_DIE_SIDES
        and action != "damage"
    ):
        await callback.answer(UNKNOWN_BUTTON_TEXT, show_alert=True)
        return
    # Гасим «часики» на кнопке — обязательно для любой CallbackQuery.
    await callback.answer()

    message, user_id = target
    session = get_session(user_id)

    if action in DAMAGE_DIE_SIDES:
        roll, label = roll_damage_die(session.character, DAMAGE_DIE_SIDES[action])
        context = roll.context_message(f"бросок урона {roll.notation} ({label})")
    elif action == "damage":
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
    # Проверка характеристики — модификатор характеристики без бонуса мастерства.
    modifier = character.roll_modifier(code)
    roll = make_roll(count=1, sides=20, modifier=modifier)

    # Служебное сообщение Мастеру с уже посчитанным итогом («кубик + мод = ИТОГ»).
    context = roll.context_message(
        f"проверка {ability_genitive} (мод. {format_modifier(modifier)})"
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


@router.callback_query(F.data.startswith("save:"))
async def handle_save_button(callback: CallbackQuery) -> None:
    """Инлайн-кнопка спасброска: d20 + характеристика + бонус мастерства (если владеет).

    Владения спасбросками определяются классом героя (PHB 2024), поэтому модификатор
    считает код — Мастер получает готовый итог и не пересчитывает его сам.
    """
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
    proficient = code in character.save_proficiencies
    modifier = character.save_modifier(code)
    roll = make_roll(count=1, sides=20, modifier=modifier)

    prof_note = "владение классом" if proficient else "без владения"
    # Служебное сообщение Мастеру с уже посчитанным итогом («кубик + мод = ИТОГ»).
    context = roll.context_message(
        f"спасбросок {ability_genitive} (мод. {format_modifier(modifier)}, {prof_note})"
    )

    logger.info(
        "Пользователь %s прошёл спасбросок %s: d20(%s)%s = %s",
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

    # Игрок ранее заявил о желании свести счёты с жизнью: ждём явного ответа на дисклеймер.
    # «да» — герой умирает и игра перезапускается; «нет» или иной ответ — партия продолжается.
    if session.pending_suicide:
        session.pending_suicide = False
        if is_suicide_confirmation(text):
            await _execute_character_selfdeath(message, session)
            return
        if is_suicide_cancellation(text):
            await message.answer(
                SUICIDE_CANCEL_TEXT,
                reply_markup=phase_keyboard(session.character),
            )
            return
        # Ответ не похож ни на «да», ни на «нет»: считаем, что игрок передумал,
        # и обрабатываем его сообщение как обычное действие ниже.

    # Игрок заговорил о суициде: сначала дисклеймер с подтверждением — гибель только по «да».
    if session.character.hero_confirmed and is_suicide_request(text):
        session.pending_suicide = True
        await message.answer(SUICIDE_CONFIRM_TEXT)
        return

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
        "быстрого броска (🎲 d20, 🎲 Бросок урона, 🧠 Проверки по статам, 📜 Заклинания) "
        "или используй команды /roll, /sheet, /inventory, /check, /spells, /rest.",
        reply_markup=build_action_keyboard(session.character),
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
    BotCommand(command="spells", description="Книга заклинаний: ячейки, применение, подготовка"),
    BotCommand(command="rest", description="Отдых: восстановить HP и ячейки заклинаний"),
]


async def main() -> None:
    """Точка входа: проверяет конфиг, поднимает polling и корректно всё закрывает."""
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit(
            "Не задан TELEGRAM_BOT_TOKEN.\n"
            "Создайте файл .env на основе .env.example и укажите токен от @BotFather."
        )
    if not LLM_API_KEY:
        raise SystemExit(
            "Не задан LLM_API_KEY.\n"
            "Создайте файл .env на основе .env.example и укажите ключ DeepSeek API."
        )

    # parse_mode=None: ответы Мастера — «сырой» текст, чтобы разметка модели
    # не ломала отправку сообщений. Промпт просит писать без Markdown.
    bot = Bot(token=TELEGRAM_BOT_TOKEN, default=DefaultBotProperties(parse_mode=None))
    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    # Белый список и ограничение частоты: применяем к сообщениям и нажатиям кнопок.
    access_middleware = AccessAndRateLimitMiddleware()
    dispatcher.message.outer_middleware(access_middleware)
    dispatcher.callback_query.outer_middleware(access_middleware)
    logger.info(
        "Доступ: %s; лимит %d запросов за %.0f с (админов: %d).",
        "белый список" if ALLOWED_USER_IDS else "все пользователи",
        RATE_LIMIT_MAX_REQUESTS,
        RATE_LIMIT_WINDOW_SECONDS,
        len(ADMIN_USER_IDS),
    )

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
        if llm_client is not None:
            await llm_client.close()
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


