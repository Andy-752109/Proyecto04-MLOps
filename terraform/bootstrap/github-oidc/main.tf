resource "aws_iam_openid_connect_provider" "github" {
  count = var.existing_oidc_provider_arn == null ? 1 : 0

  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
}

locals {
  oidc_provider_arn = var.existing_oidc_provider_arn != null ? var.existing_oidc_provider_arn : aws_iam_openid_connect_provider.github[0].arn
}

# Read-only access to the DVC remote so CI can run the PROD quality gate.
resource "aws_iam_role" "github_actions" {
  name                 = "mlops-p2-github-oidc"
  description          = "GitHub OIDC authentication check for the main branch of the MLOps project"
  max_session_duration = 3600

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = local.oidc_provider_arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          "token.actions.githubusercontent.com:sub" = [
            "repo:karenelizabg@153574034/proyecto-fase3-MLOPS@1381627489:ref:refs/heads/main",
            "repo:karenelizabg@153574034/proyecto-fase3-MLOPS@1381627489:pull_request",
          ]
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "dvc_read" {
  name = "dvc-remote-read"
  role = aws_iam_role.github_actions.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "s3:ListBucket"
        Resource = "arn:aws:s3:::${var.dvc_bucket_name}"
      },
      {
        Effect   = "Allow"
        Action   = "s3:GetObject"
        Resource = "arn:aws:s3:::${var.dvc_bucket_name}/*"
      },
    ]
  })
}
