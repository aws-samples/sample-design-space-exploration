/**
 * Runtime configuration for Car Design Explorer.
 * 
 * In production, /config.json is deployed by CDK with the correct values.
 * For local dev, edit the values below directly.
 */

// Default config — overridden at runtime by /config.json
let runtimeConfig = null;

export async function loadConfig() {
  try {
    const resp = await fetch('/config.json');
    if (resp.ok) {
      runtimeConfig = await resp.json();
    }
  } catch {
    // config.json not available — use defaults below
  }
  return getConfig();
}

export function getConfig() {
  if (runtimeConfig) {
    return {
      Auth: {
        Cognito: {
          userPoolId: runtimeConfig.cognito?.userPoolId || '',
          userPoolClientId: runtimeConfig.cognito?.userPoolClientId || '',
          region: runtimeConfig.cognito?.region || 'us-east-1',
        }
      },
      websocketUrl: runtimeConfig.websocket?.url || ''
    };
  }
  // Fallback for local dev
  return {
    Auth: {
      Cognito: {
        userPoolId: process.env.REACT_APP_COGNITO_USER_POOL_ID || '',
        userPoolClientId: process.env.REACT_APP_COGNITO_CLIENT_ID || '',
        region: process.env.REACT_APP_AWS_REGION || 'us-east-1',
      }
    },
    websocketUrl: process.env.REACT_APP_WEBSOCKET_URL || ''
  };
}
