"""Auth Stack — Cognito user pool, identity pool, and app client.

Self-signup is disabled; deployment operators provision the frontend user.
"""

import aws_cdk as cdk
from aws_cdk import aws_cognito as cognito
from constructs import Construct


class AuthStack(cdk.Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        self.user_pool = cognito.UserPool(
            self, "CarDesignUserPool",
            user_pool_name="CarDesignExplorerUsers",
            self_sign_up_enabled=False,
            sign_in_aliases=cognito.SignInAliases(username=True, email=True),
            auto_verify=cognito.AutoVerifiedAttrs(email=True),
            standard_attributes=cognito.StandardAttributes(
                email=cognito.StandardAttribute(required=True, mutable=True),
            ),
            password_policy=cognito.PasswordPolicy(
                min_length=8,
                require_lowercase=True,
                require_uppercase=True,
                require_digits=True,
            ),
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )

        # Force AllowAdminCreateUserOnly=True so only admins can create users.
        # CDK's self_sign_up_enabled doesn't always flip this on updates.
        cfn_pool = self.user_pool.node.default_child
        cfn_pool.add_property_override(
            "AdminCreateUserConfig.AllowAdminCreateUserOnly", True
        )

        self.app_client = self.user_pool.add_client(
            "CarDesignAppClient",
            user_pool_client_name="car-design-explorer-web",
            auth_flows=cognito.AuthFlow(
                user_password=True,
                user_srp=True,
            ),
            generate_secret=False,
        )

        self.identity_pool = cognito.CfnIdentityPool(
            self, "CarDesignIdentityPool",
            identity_pool_name="CarDesignExplorerIdentity",
            allow_unauthenticated_identities=False,
            cognito_identity_providers=[
                cognito.CfnIdentityPool.CognitoIdentityProviderProperty(
                    client_id=self.app_client.user_pool_client_id,
                    provider_name=self.user_pool.user_pool_provider_name,
                )
            ],
        )

        cdk.CfnOutput(self, "UserPoolId", value=self.user_pool.user_pool_id)
        cdk.CfnOutput(self, "UserPoolClientId", value=self.app_client.user_pool_client_id)
        cdk.CfnOutput(self, "IdentityPoolId", value=self.identity_pool.ref)
