import React, { useState } from 'react';
import { signIn } from 'aws-amplify/auth';
import './Login.css';

const Login = ({ onSignIn }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleLogin = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    try {
      const result = await signIn({ username, password });
      if (result.nextStep?.signInStep === 'CONFIRM_SIGN_IN_WITH_NEW_PASSWORD_REQUIRED') {
        const { confirmSignIn } = await import('aws-amplify/auth');
        await confirmSignIn({ challengeResponse: password });
      }
      onSignIn({ username });
    } catch (err) {
      setError(err.message || 'Sign in failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-background">
        <div className="login-card">
          <div className="login-header">
            <div className="aws-logo">
              <div className="aws-icon">🏎️</div>
            </div>
            <h1 className="login-title">Car Design Explorer</h1>
            <p className="login-subtitle">Sign in to continue</p>
          </div>

          {error && <div className="error-message">{error}</div>}

          <form onSubmit={handleLogin} className="login-form">
              <div className="form-group">
                <label htmlFor="username" className="form-label">Username</label>
                <input type="text" id="username" className="form-input" placeholder="Enter your username"
                  value={username} onChange={(e) => setUsername(e.target.value)} required disabled={isLoading} />
              </div>
              <div className="form-group">
                <label htmlFor="password" className="form-label">Password</label>
                <div className="password-input-container">
                  <input type={showPassword ? 'text' : 'password'} id="password"
                    className="form-input password-input" placeholder="Enter your password"
                    value={password} onChange={(e) => setPassword(e.target.value)} required disabled={isLoading} />
                  <button type="button" className="password-toggle"
                    onClick={() => setShowPassword(!showPassword)} disabled={isLoading}>
                    {showPassword ? '🙈' : '👁️'}
                  </button>
                </div>
              </div>
              <button type="submit" className="sign-in-button" disabled={isLoading || !username || !password}>
                {isLoading ? 'Signing in...' : 'Sign in'}
              </button>
            </form>

          <div className="login-footer">
            <p className="copyright">Powered by Amazon Bedrock AgentCore</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
