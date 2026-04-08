# PilotAI Strategy Recommendation API

Retrieve portfolio weightage and stock allocations for investment strategies.

## Endpoint

```
POST https://ai-stag.pilotai.com/v2/strategy_recommendation
```

## Authentication

Include the API key in the `x-api-key` header:

```
x-api-key: cZZP6he1Qez8Lb6njh6w5vUe
```

## Request

### Headers

| Header | Value | Required |
|--------|-------|----------|
| `Content-Type` | `application/json` | Yes |
| `x-api-key` | `cZZP6he1Qez8Lb6njh6w5vUe` | Yes |

### Body

```json
{
  "strategy_slugs": ["dividend-aristocrats", "growth-investing"]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `strategy_slugs` | string[] | List of strategy slugs to retrieve. If omitted or empty, returns top recommendations only. |

### Example (curl)

```bash
# Single strategy
curl -s -X POST https://ai-stag.pilotai.com/v2/strategy_recommendation \
  -H "Content-Type: application/json" \
  -H "x-api-key: cZZP6he1Qez8Lb6njh6w5vUe" \
  -d '{"strategy_slugs": ["dividend-aristocrats"]}'

# Multiple strategies
curl -s -X POST https://ai-stag.pilotai.com/v2/strategy_recommendation \
  -H "Content-Type: application/json" \
  -H "x-api-key: cZZP6he1Qez8Lb6njh6w5vUe" \
  -d '{"strategy_slugs": ["dividend-aristocrats", "growth-investing", "low-beta-stocks"]}'

# Top recommendations only (no specific slugs)
curl -s -X POST https://ai-stag.pilotai.com/v2/strategy_recommendation \
  -H "Content-Type: application/json" \
  -H "x-api-key: cZZP6he1Qez8Lb6njh6w5vUe" \
  -d '{}'
```

## Response

Returns a JSON object with two arrays:

```json
{
  "user_recommendation": [
    {
      "strategy_slug": "dividend-aristocrats",
      "strategy_name": "Dividend Aristocrats",
      "strategy_tag": ["Dividends", "Defensive", "Multi Sector"],
      "strategy_definition": "Invest in companies with decades of consistent & increasing dividend payments.",
      "total_cost": 89155.98,
      "leftover": 10844.02,
      "candidate_asset": [
        {
          "ticker": "T",
          "name": "AT&T Inc.",
          "price": 28.65,
          "quantity": 134,
          "weights": 0.0428,
          "cost": 3839.10
        }
      ],
      "stock_score": {
        "value": 2.85,
        "growth": 1.92,
        "health": 3.41,
        "momentum": 2.67,
        "past_performance": 2.13
      }
    }
  ],
  "top_recommendation": [
    // system-curated top strategies (always returned, same structure)
  ]
}
```

### Response Fields

| Field | Description |
|-------|-------------|
| `user_recommendation` | Strategies matching the `strategy_slugs` you requested. Empty array if none provided. |
| `top_recommendation` | System-curated top strategies. Always returned regardless of input. |
| `strategy_slug` | Unique identifier for the strategy. |
| `strategy_name` | Display name of the strategy. |
| `strategy_tag` | Category tags (e.g. "Dividends", "Growth", "Defensive"). |
| `total_cost` | Total cost of the portfolio allocation (based on $100,000 budget). |
| `leftover` | Remaining cash after allocation. |
| `candidate_asset` | Array of stocks with ticker, name, price, quantity, weight, and cost. |
| `candidate_asset[].weights` | Portfolio weight of this stock (decimal, e.g. 0.0428 = 4.28%). |
| `stock_score` | Weighted portfolio-level scores across value, growth, health, momentum, and past performance. |

## Caching

- **First request:** May take 1-2 seconds (fetches live prices, calculates optimal allocations).
- **Subsequent requests:** Served from cache (near-instant) for 30 minutes. Cache resets daily (US/Eastern timezone).
- Cache key is based on the combination of strategy slugs requested.

## Available Strategy Slugs

```
5g-infrastructure                    aging-population
ai-related-companies                 biomedical-and-genetics-industry
biotech-breakthroughs                buffett-bargains
clean-energy-revolution              cloud-computing-boom
consumer-discretionary               consumer-staples-stability-strategy
contrarian-investing                 cybersecurity-shield
deep-value-investing                 defensive-investing
diversified-bluechips                dividend-aristocrats
drip-dividend-reinvestment-plan      e-commerce-enablers
electric-vehicle-ev-boom             energy-sector-growth-strategy
esg-leaders                          fallen-angels
financials-sector-capital-strategy   gaming-giants
global-investing                     gold-mining-industry
green-infrastructure                 growth-investing
healthcare-sector-stability-and-growth-fund
high-beta-stocks                     high-dividend-stocks
industrials-sector-infrastructure-fund
investment-management-industry
leisure-and-recreation-services-industry
low-beta-stocks                      low-volatility-stocks
manufacturing-industry               market-disruptors
meme-stock-mania                     metaverse-pioneers
mid-cap-stocks                       momentum-investing
quality-investing                    real-estate-sector-income-fund
robotics-automation                  sector-rotation
semiconductor-supercycle             small-cap-stocks
socially-responsible-investing-sri   space-exploration
technology-sector-innovation-fund    the-amazon-of-x
thematic-investing                   transportation-airline-sector
utilities-sector-stability-fund      value-investing
water-scarcity-solutions
```

> **Note:** Strategy slugs are sourced from the database and may be updated over time. If a slug is not found, the endpoint will still return `top_recommendation` but `user_recommendation` may be empty.
