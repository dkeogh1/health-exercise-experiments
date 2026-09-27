# Strava OAuth Setup Guide

## Step 1: Create a Strava App

1. Go to https://www.strava.com/settings/api
2. Click "Create an Application"
3. Fill in:
   - **Application name:** `strava-analysis` (or your choice)
   - **Category:** Data Analysis
   - **Website:** `http://localhost` (for local dev)
   - **Authorization callback domain:** `localhost`
4. Click "Create"

## Step 2: Get Your Credentials

After creating the app, you'll see:
- **Client ID**
- **Client Secret**

Copy these to `config/.env`:
```
STRAVA_CLIENT_ID=your_client_id
STRAVA_CLIENT_SECRET=your_client_secret
```

## Step 3: Run the OAuth Flow

```bash
python scripts/strava_auth.py
```

This will:
1. Open your browser to Strava's authorization page
2. Ask you to approve the app (scopes: `activity:read_all`)
3. Save your access token + refresh token to `config/.env`

On a headless host (dkbl1), use `python scripts/strava_auth_manual.py`
instead: it prints the URL to open on your own machine and asks you to
paste the code back. `strava_auth.py` needs a local browser.

## Step 4: Verify

Run:
```bash
python scripts/strava_export.py
```

Should write `data/raw/activities_<stamp>.json` + `.csv`. If it works, you're good to go.

## Token Refresh

`strava_export.py` and `strava_streams.py` refresh the access token automatically. If you still get auth errors, run the auth script again.

## Scopes

Currently requesting:
- `activity:read_all` — Read all activities (private + public)

Other available scopes (if needed later):
- `athlete:read_all` — Full athlete profile
- `activity:write` — Create/modify activities
