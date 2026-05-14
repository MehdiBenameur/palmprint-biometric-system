namespace Palmprint.Application.DTOs;

public class IdentifyCandidateDto
{
    public Guid UserId { get; set; }

    public string FullName { get; set; } = string.Empty;

    public string ExternalId { get; set; } = string.Empty;

    public double Score { get; set; }
}