# WhatsApp Cloud API Setup Guide

Step-by-step guide to connecting FrankClaw to WhatsApp via the Meta Business Platform.
This was tested end-to-end on 2026-03-12.

## Prerequisites

- A Meta (Facebook) account
- A phone with WhatsApp installed (your personal number, for receiving test messages)
- FrankClaw built: `cargo build -p frankclaw`
- A public URL for webhooks (Cloudflare Tunnel, ngrok, or similar)

## Step 1: Create a Meta App

1. Go to [developers.facebook.com](https://developers.facebook.com)
2. Click **Create App** → choose **Business** type
3. Give it a name (e.g., "frankclaw") and select your business account
4. Once created, add the **WhatsApp** product from the app dashboard

## Step 2: Privacy Policy and Terms of Service

Meta requires a privacy policy and terms of service URL before the app can go live.
FrankClaw includes templates at:

- `docs/PRIVACY_POLICY.md`
- `docs/TERMS_OF_SERVICE.md`

Host these anywhere publicly accessible (GitHub Pages, your domain, etc.) and add
the URLs in **App Settings → Basic → Privacy Policy URL / Terms of Service URL**.

Then set the app to **Live** mode (toggle at the top of the dashboard).

## Step 3: Create a System User and Generate a Token

The temporary token from the API Setup page expires in 24 hours. For a permanent token:

1. Go to [business.facebook.com](https://business.facebook.com) → **Settings** → **Users** → **System Users**
2. Click **Add** → name it (e.g., "FrankClaw") → choose **Admin** role
3. **Before generating a token**, you must assign the app as a role:
   - Go to [developers.facebook.com](https://developers.facebook.com) → Your App → **App Roles** → **Roles**
   - Click **Add People** or **Add System Users**
   - Search for your system user and assign the **Developer** or **Admin** role
4. Go back to **Business Settings → System Users** → select your system user
5. Click **Add Assets** → **Apps** tab → select your app → grant **Full Control**
6. Also under **Add Assets** → **WhatsApp Accounts** → select your WhatsApp Business Account
7. Click **Generate New Token** → select your app
8. Grant permissions: `whatsapp_business_messaging`, `whatsapp_business_management`
9. Copy the token — this one does not expire

**Common issue:** If you see "Nenhuma permissão disponível" (no permissions available) when
generating the token, it means the system user hasn't been assigned a role on the app yet.
Go back to step 3 above.

**Common issue:** If the system user token returns "Object does not exist" errors when calling
the API, the WhatsApp Business Account asset wasn't assigned. Go back to step 6.

## Step 4: Collect Your Credentials

You need four values:

| Credential | Where to find it |
|-----------|-----------------|
| **Access Token** | Generated in Step 3 (or use the temporary one from API Setup for testing) |
| **Phone Number ID** | [developers.facebook.com](https://developers.facebook.com) → Your App → **WhatsApp** → **API Setup** → shown under the test phone number |
| **Verify Token** | Any string you choose — used to verify webhook registration |
| **App Secret** | [developers.facebook.com](https://developers.facebook.com) → Your App → **Settings** → **Basic** → click **Show** next to App Secret |

## Step 5: Set Environment Variables

Add to your shell secrets file (e.g., `~/.config/zsh/secrets` or `~/.bashrc`):

```bash
export WHATSAPP_ACCESS_TOKEN="EAAxxxxxxx..."
export WHATSAPP_PHONE_NUMBER_ID="1024468327416362"
export WHATSAPP_VERIFY_TOKEN="any-random-string-you-pick"
export WHATSAPP_APP_SECRET="abcdef123456"
```

**Important:** The config file (`frankclaw.toml`) stores environment variable *names*, not
the actual secret values. For example, `"access_token_env": "WHATSAPP_ACCESS_TOKEN"` tells
FrankClaw to read the value from the `WHATSAPP_ACCESS_TOKEN` environment variable at startup.
Do not paste the actual token into the JSON config.

## Step 6: Configure FrankClaw

```bash
source ~/.config/zsh/secrets
./target/debug/frankclaw onboard --channel whatsapp
./target/debug/frankclaw check
./target/debug/frankclaw doctor
```

This writes the WhatsApp channel config to `frankclaw.toml`. Verify it looks like:

```json
{
  "channels": {
    "whatsapp": {
      "enabled": true,
      "accounts": [
        {
          "access_token_env": "WHATSAPP_ACCESS_TOKEN",
          "phone_number_id_env": "WHATSAPP_PHONE_NUMBER_ID",
          "verify_token_env": "WHATSAPP_VERIFY_TOKEN",
          "app_secret_env": "WHATSAPP_APP_SECRET"
        }
      ]
    }
  }
}
```

## Step 7: Set Up a Public URL

WhatsApp sends webhooks to a public HTTPS URL. Choose one:

### Cloudflare Tunnel (recommended for stable setups)

```bash
# Install
sudo pacman -S cloudflared   # Arch
# or: brew install cloudflared

# Login (one-time, opens browser)
cloudflared tunnel login

# Create tunnel and DNS route
cloudflared tunnel create frankclaw
cloudflared tunnel route dns frankclaw claw.yourdomain.com
```

Create `~/.cloudflared/config.yml`:

```yaml
tunnel: frankclaw
credentials-file: /home/youruser/.cloudflared/<tunnel-id>.json

ingress:
  - hostname: claw.yourdomain.com
    service: http://localhost:18789
  - service: http_status:404
```

Run:

```bash
cloudflared tunnel run frankclaw
```

### ngrok (quick testing)

```bash
ngrok http 18789
```

Note: the ngrok URL changes every restart, so you'll need to re-register the
webhook each time. Cloudflare Tunnel gives you a stable URL.

## Step 8: Register the Webhook in Meta

1. Go to [developers.facebook.com](https://developers.facebook.com) → Your App → **WhatsApp** → **Configuration**
2. Under **Webhook**, click **Edit**
3. **Callback URL**: `https://claw.yourdomain.com/api/whatsapp/webhook`
4. **Verify token**: the same string you set in `WHATSAPP_VERIFY_TOKEN`
5. Click **Verify and Save** — Meta sends a GET request to your URL, FrankClaw responds
6. Under **Webhook fields**, click **Subscribe** next to **messages**

**You must subscribe to the `messages` field.** Without this, Meta will not forward
incoming messages to your webhook, even though the URL is verified.

## Step 9: Add Test Recipients

Until your app passes Meta's business verification, you can only exchange messages
with numbers you've explicitly registered:

1. Go to **WhatsApp** → **API Setup**
2. Under the **"To"** field, click **Manage phone number list**
3. Add your personal WhatsApp number (with country code, e.g., `5511999998888`)
4. You'll receive a verification code on WhatsApp — enter it to confirm

## Step 10: Start the Gateway

```bash
source ~/.config/zsh/secrets
./target/debug/frankclaw gateway
```

## Step 11: Send the First Message

You cannot message the test phone number directly. The flow is:

1. **Send a template message from Meta to your phone** — go to **API Setup** and click
   **Send message** (or use curl):

```bash
curl -X POST "https://graph.facebook.com/v21.0/$WHATSAPP_PHONE_NUMBER_ID/messages" \
  -H "Authorization: Bearer $WHATSAPP_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messaging_product": "whatsapp",
    "to": "YOUR_PHONE_WITH_COUNTRY_CODE",
    "type": "template",
    "template": {
      "name": "hello_world",
      "language": { "code": "en_US" }
    }
  }'
```

2. **Receive the template message** on your WhatsApp
3. **Reply to it** — this reply hits your webhook and FrankClaw processes it

The WhatsApp Business API requires a business-initiated template message to open a
conversation. After that, you have a **24-hour window** for free-form messaging.

## Step 12: Approve Pairing

The first time a new sender messages FrankClaw, it creates a **pairing request**
(this is a security feature). You'll see in the gateway logs:

```
INFO security audit event action="pairing.pending" outcome="created"
```

List and approve the pairing:

```bash
./target/debug/frankclaw pairing-list
# Note the pairing code from the output
./target/debug/frankclaw pairing-approve whatsapp <PAIRING_CODE>
```

After approving, send another message from your phone. This time FrankClaw will
forward it to the AI provider and send the response back.

## Troubleshooting

### Message sent via API but not received on phone

- **Wait a few minutes.** Meta's test infrastructure can be slow to provision for new apps.
- Check WhatsApp → Settings → Privacy → ensure "Block unknown contacts" is off.
- Check message requests / spam folder in WhatsApp.
- Verify the phone number format: country code + number, no `+`, no dashes (e.g., `5511999998888`).

### Webhook verified but no inbound messages

- Did you **subscribe to the `messages` webhook field**? Verification alone is not enough.
- Is the gateway actually running? Test: `curl https://your-url/api/whatsapp/webhook` should
  return a non-404 response.

### "Object does not exist" or 401 errors from the API

- The access token may not have permission on the WhatsApp Business Account.
- If using a system user token: make sure the WhatsApp Business Account is assigned as an
  asset to the system user (Business Settings → System Users → Add Assets → WhatsApp Accounts).
- Try the temporary token from the API Setup page to confirm the phone number ID is correct.

### "Provider references missing environment variable"

- The `frankclaw.toml` config expects **environment variable names**, not values.
- Wrong: `"api_key_ref": "sk-proj-abc123..."` (actual key)
- Right: `"api_key_ref": "OPENAI_API_KEY"` (env var name)
- Make sure you `source` your secrets file before starting the gateway.

### Pairing pending — messages received but no response

- Run `./target/debug/frankclaw pairing-list` to see pending requests.
- Approve with `./target/debug/frankclaw pairing-approve whatsapp <CODE>`.
- Restart the gateway and try again.

### Model provider errors (401, invalid API key)

- Verify your AI provider API key is valid and has billing set up.
- OpenRouter keys (`sk-or-...`) need `"base_url": "https://openrouter.ai/api/v1"` in the
  provider config — they don't work against `api.openai.com`.
- Check with `./target/debug/frankclaw doctor` for provider connectivity issues.

## Architecture Notes

- **Webhook endpoint:** `GET /api/whatsapp/webhook` for Meta's verification challenge,
  `POST /api/whatsapp/webhook` for incoming messages.
- **Signature verification:** When `WHATSAPP_APP_SECRET` is set, FrankClaw verifies the
  `X-Hub-Signature-256` HMAC-SHA256 header on all incoming webhooks.
- **24-hour messaging window:** WhatsApp requires a template message to initiate contact.
  After the user replies, free-form messages can be exchanged for 24 hours.
- **DM pairing:** FrankClaw blocks messages from unknown senders by default. Each new
  sender must be approved via `pairing-approve` before their messages are processed.
