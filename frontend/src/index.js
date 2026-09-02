import React from 'react';
import ReactDOM from 'react-dom/client';
import { Amplify } from 'aws-amplify';
import App from './App';
import { loadConfig } from './aws-config';

loadConfig().then((cfg) => {
  Amplify.configure({
    Auth: {
      Cognito: {
        userPoolId: cfg.Auth.Cognito.userPoolId,
        userPoolClientId: cfg.Auth.Cognito.userPoolClientId,
      },
    },
  });

  const root = ReactDOM.createRoot(document.getElementById('root'));
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
});
