namespace Palmprint.Application.DTOs;

public class IdentifyResponse
{
    public bool Success { get; set; }

    public Guid? UserId { get; set; }

    public string? FullName { get; set; }

    public string? ExternalId { get; set; }

    public double SimilarityScore { get; set; }

    public string Message { get; set; } = string.Empty;

    public List<IdentifyCandidateDto> TopCandidates { get; set; } = new();
}