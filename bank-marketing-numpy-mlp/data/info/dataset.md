# Bank Marketing

- **Источник:** UCI Machine Learning Repository, <https://archive.ics.uci.edu/dataset/222/bank+marketing>
- **Как получен файл:** архив `https://archive.ics.uci.edu/static/public/222/bank+marketing.zip` (1 МБ), внутри `bank.zip`, из него взяты `bank-full.csv` и `bank-names.txt`.
- **Авторы:** S. Moro, P. Rita, P. Cortez (2014), DOI 10.24432/C5K306. Работа с описанием: Moro, Laureano, Cortez, ESM 2011.
- **Лицензия:** CC BY 4.0.
- **Одна строка:** один клиент в кампании телефонного маркетинга португальского банка. Строки идут по порядку дат, май 2008 — ноябрь 2010.
- **Размер:** 45 211 строк, 16 признаков и цель `y`.

Полное описание от авторов: [../raw/bank-names.txt](../raw/bank-names.txt).

| Столбец | Смысл | Что сделано в работе |
|---|---|---|
| `age` | возраст | стандартизация, флаг `is_young_or_senior` |
| `job`, `marital`, `education` | профессия, семейное положение, образование | one-hot, `unknown` как отдельная категория |
| `default`, `housing`, `loan` | дефолт, ипотека, потребкредит | one-hot, флаг `has_any_loan` |
| `balance` | средний годовой баланс, евро | знаковый логарифм, флаг `is_negative_balance` |
| `contact` | тип связи | one-hot |
| `day`, `month` | день и месяц последнего контакта | `day` число, `month` one-hot |
| `duration` | длительность звонка | **удалён: утечка** |
| `campaign` | звонков в этой кампании, включая последний | логарифм |
| `pdays` | дней после прошлого контакта, −1 = не звонили | флаг `was_contacted_before`, признак `recency` |
| `previous` | звонков раньше | логарифм |
| `poutcome` | итог прошлой кампании | one-hot |
| `y` | открыл ли депозит | цель, 1 = да |

Сырые файлы в `data/raw/` не изменяются. Преобразования применяются кодом из `src/features.py` при каждом запуске, поэтому `data/processed/` не нужна.
