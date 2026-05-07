namespace Palmprint.Application.DTOs;

public class EnrollResponse
{
    public Guid UserId { get; set; }
    public Guid TemplateId { get; set; }
    public double QualityScore { get; set; }
    public string Message { get; set; } = string.Empty;
}